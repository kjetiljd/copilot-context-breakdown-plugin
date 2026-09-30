#!/usr/bin/env python3
"""Show the content mix of the current GitHub Copilot CLI session's context window.

Run it from Copilot CLI (`!breakdown.py`), or use the /context-breakdown skill
from the context-breakdown plugin.
The session is found via COPILOT_AGENT_SESSION_ID, or else via the inuse lock
of the CLI process the script runs under.

Sources:
  * /context data: the extension in the context-breakdown plugin fetches the
    CLI's own token counts (the same numbers as /context) - system prompt,
    custom instructions, tool definitions and tokens for every message in the
    context - and writes them to <session-state>/<id>/context-breakdown.json.
  * events.jsonl: messages are matched against the events in the session log
    so that each message can be split into categories (user message, tool
    call, tool result, model output, reasoning ...). Only the split *within* a
    message is estimated; the total per message is the CLI's. The log also
    provides the "input summed over all calls".

Without the extension the script falls back to pure estimates from events.jsonl.

Examples:
  breakdown.py
  breakdown.py --by-tool --defs --heaviest 10
  breakdown.py --json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = Path(os.environ.get("COPILOT_SESSION_STATE", "~/.copilot/session-state")).expanduser()
SNAPSHOT_FILE = "context-breakdown.json"

CATEGORIES = [
    ("system", "System prompt"),
    ("custom", "Custom instructions"),
    ("tooldefs", "Tool definitions"),
    ("skills", "Skills & instruction files"),
    ("reminders", "System reminders & notifications"),
    ("user", "User messages"),
    ("tools", "Tool calls incl results"),
    ("output", "Model outputs"),
    ("reasoning", "Reasoning (visible)"),
    ("summary", "Compaction summaries"),
]
SUBROWS = {
    "tooldefs": [("mcptools", "of which MCP")],
    "tools": [("toolcall", "calls (arguments)"), ("toolresult", "results")],
}
# Keys that add up to the total ("tools" = toolcall + toolresult, "mcptools" is a subset).
ADDITIVE = ["system", "custom", "tooldefs", "skills", "reminders", "user", "toolcall", "toolresult", "output", "reasoning", "summary"]
MESSAGE_KEYS = ["skills", "reminders", "user", "toolcall", "toolresult", "output", "reasoning", "summary"]

# Heuristic when tiktoken is missing. Measured against the CLI's count (Claude): prose
# and reasoning ~3.5-4 chars/token, JSON-escaped tool arguments ~2-2.5, and each
# tool call costs ~20 extra tokens of structure.
DEFAULT_CHARS_PER_TOKEN = 4.0
ARGS_CHARS_PER_TOKEN = 2.5
CALL_OVERHEAD = 20
IMAGE_TOKENS_MAX = 1600


# ---------------------------------------------------------------- tokens

class TokenCounter:
    def __init__(self, chars_per_token: float, use_tiktoken: bool = True):
        self.cpt = chars_per_token
        self.enc = None
        if use_tiktoken:
            try:
                import tiktoken  # type: ignore

                self.enc = tiktoken.get_encoding("o200k_base")
            except Exception:
                self.enc = None

    @property
    def method(self) -> str:
        return "tiktoken o200k_base" if self.enc else f"~{self.cpt:g} chars/token, ~{ARGS_CHARS_PER_TOKEN:g} for arguments"

    def __call__(self, text: str | None) -> int:
        if not text:
            return 0
        if self.enc:
            return len(self.enc.encode(text, disallowed_special=()))
        return round(len(text) / self.cpt)

    def args(self, text: str | None) -> int:
        if not text:
            return 0
        if self.enc:
            return len(self.enc.encode(text, disallowed_special=()))
        return round(len(text) / ARGS_CHARS_PER_TOKEN)


def image_dims(raw: bytes) -> tuple[int, int] | None:
    if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 24:
        return struct.unpack(">II", raw[16:24])
    if raw[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(raw):
            if raw[i] != 0xFF:
                i += 1
                continue
            if raw[i + 1] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", raw[i + 5 : i + 9])
                return w, h
            i += 2 + struct.unpack(">H", raw[i + 2 : i + 4])[0]
    return None


def image_tokens(dims: tuple[int, int] | None) -> int:
    """Anthropic-style estimate: w*h/750 after downscaling to at most 1568 px."""
    if not dims:
        return IMAGE_TOKENS_MAX
    w, h = dims
    scale = min(1.0, 1568 / max(w, h, 1))
    return max(1, min(IMAGE_TOKENS_MAX, round(w * scale * h * scale / 750)))


# ---------------------------------------------------------------- session

def parent_pid(pid: int) -> int:
    try:
        out = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip() or 0)
    except Exception:
        return 0


def current_session() -> Path | None:
    sid = os.environ.get("COPILOT_AGENT_SESSION_ID")
    if sid and (STATE_DIR / sid / "events.jsonl").exists():
        return STATE_DIR / sid
    pid, seen = os.getppid(), set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        dirs = [lock.parent for lock in STATE_DIR.glob(f"*/inuse.{pid}.lock") if (lock.parent / "events.jsonl").exists()]
        if dirs:
            return max(dirs, key=lambda p: (p / "events.jsonl").stat().st_mtime)
        pid = parent_pid(pid)
    return None


def resolve_session(arg: str | None) -> Path:
    if not arg:
        found = current_session()
        if not found:
            sys.exit(
                "Could not find the current Copilot CLI session. Run the script from Copilot CLI "
                "(e.g. with the ! prefix), or pass --session <id>."
            )
        return found
    p = Path(arg).expanduser()
    if p.is_file():
        return p.parent
    if p.is_dir():
        return p
    matches = sorted(STATE_DIR.glob(f"{arg}*/events.jsonl"))
    if len(matches) == 1:
        return matches[0].parent
    if not matches:
        sys.exit(f"No session matching '{arg}' in {STATE_DIR}")
    sys.exit(f"'{arg}' is ambiguous: " + ", ".join(m.parent.name for m in matches[:5]))


def read_events(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # the last line may be half-written in an active session


def workspace_field(session_dir: Path, key: str) -> str:
    ws = session_dir / "workspace.yaml"
    if ws.exists():
        for line in ws.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(f"{key}:"):
                return line.split(":", 1)[1].strip()
    return ""


def load_snapshot(session_dir: Path) -> dict | None:
    try:
        data = json.loads((session_dir / SNAPSHOT_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("sessionId") != session_dir.name or not data.get("contextAttribution"):
        return None
    return data


# ---------------------------------------------------------------- messages from events.jsonl

@dataclass(eq=False)
class Message:
    role: str  # user | assistant | tool (the same roles /context uses)
    label: str = ""
    alt_label: str = ""  # tool results: /context uses either the description or the content
    parts: dict[str, float] = field(default_factory=dict)
    tool: str | None = None  # for tool results
    call_tools: dict[str, float] = field(default_factory=dict)  # for assistant messages with calls
    start: int = 0  # number of model calls before the message entered the context
    end: int | None = None  # number of model calls when it was removed (compaction)
    factor: float = 1.0  # estimate -> CLI tokens
    in_context: bool = True  # False if the CLI does not have the message in its context

    def add(self, cat: str, n: float) -> None:
        if n:
            self.parts[cat] = self.parts.get(cat, 0) + n

    @property
    def total(self) -> float:
        return sum(self.parts.values())


@dataclass
class Session:
    session_dir: Path
    name: str = ""
    model: str = ""
    system_tokens: int = 0
    system_cumulative: int = 0
    tooldefs_reported: int = 0
    messages: list[Message] = field(default_factory=list)  # in the context now
    history: list[Message] = field(default_factory=list)  # all, for the sum over calls
    model_calls: int = 0
    compactions: int = 0
    images: int = 0
    attachments: int = 0
    events_after_mark: int = 0
    mark: list[Message] | None = None
    mark_system: int = 0

    def append(self, m: Message) -> None:
        m.start = self.model_calls
        self.messages.append(m)
        self.history.append(m)


def analyse(session_dir: Path, count: TokenCounter, include_reasoning: bool, mark_id: str | None) -> Session:
    s = Session(session_dir=session_dir, name=workspace_field(session_dir, "name"))
    pending: dict[str, tuple[str, str]] = {}  # toolCallId -> (tool, label)
    assets: dict[str, tuple[int, int] | None] = {}
    skills: dict[str, str] = {}
    compaction_mark: int | None = None
    last_call_id: str | None = None
    current_assistant: Message | None = None

    for e in read_events(session_dir / "events.jsonl"):
        t = e.get("type", "")
        d = e.get("data") or {}
        if s.mark is not None:
            s.events_after_mark += 1
        if d.get("parentToolCallId"):
            continue  # subagent events belong to another context

        if t == "session.binary_asset" and str(d.get("mimeType", "")).startswith("image/") and d.get("data"):
            try:
                assets[d["assetId"]] = image_dims(base64.b64decode(d["data"][:200_000]))
            except Exception:
                assets[d["assetId"]] = None

        elif t in ("session.start", "session.resume"):
            s.model = d.get("selectedModel") or s.model
        elif t == "session.model_change":
            s.model = d.get("newModel") or s.model

        elif t == "system.message":
            s.system_tokens = count(d.get("content"))

        elif t == "user.message":
            source = d.get("source") or "user"
            content = d.get("content") or ""
            full = d.get("transformedContent") or content
            m = Message(role="user", label=full)
            if source.startswith("skill-") or source == "instruction-discovery":
                m.add("skills", count(full))
            elif source in ("system", "autopilot") or not content:
                m.add("reminders", count(full))
            elif content in full:
                head, _, tail = full.partition(content)
                m.add("user", count(content))
                m.add("reminders", count(head) + count(tail))
            else:
                m.add("user", count(full))
            s.attachments += len(d.get("attachments") or [])
            s.append(m)
            current_assistant = None

        elif t == "skill.invoked":
            skills[d.get("name") or ""] = d.get("content") or ""

        elif t == "skill.context_delivered_ref":
            # Newer CLI versions store the skill context as a reference instead of a user.message.
            name = str(d.get("source") or "").removeprefix("skill-")
            m = Message(role="user", label=f"skill: {name}")
            m.add("skills", count(d.get("prefix")) + count(skills.get(name)))
            s.append(m)
            current_assistant = None

        elif t == "system.notification":
            m = Message(role="user", label=d.get("content") or "")
            m.add("reminders", count(d.get("content")))
            s.append(m)

        elif t == "assistant.message":
            call_id = d.get("apiCallId") or d.get("messageId")
            new_call = call_id is None or call_id != last_call_id
            if new_call:
                s.model_calls += 1
                s.system_cumulative += s.system_tokens
            last_call_id = call_id
            s.model = d.get("model") or s.model
            reqs = d.get("toolRequests") or []
            if new_call or current_assistant is None:
                current_assistant = Message(role="assistant")
                s.append(current_assistant)
            m = current_assistant
            content = d.get("content") or ""
            if content and not m.label:
                m.label = content
            m.add("output", count(content))
            if include_reasoning:
                m.add("reasoning", count(d.get("reasoningText")))
            for req in reqs:
                name = req.get("name") or "?"
                args = req.get("arguments")
                args_text = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                n = count.args(name) + count.args(args_text) + CALL_OVERHEAD
                m.add("toolcall", n)
                m.call_tools[name] = m.call_tools.get(name, 0) + n
                desc = args.get("description") if isinstance(args, dict) else None
                pending[req.get("toolCallId", "")] = (name, desc or "")
            if not m.label and m.call_tools:
                names = list(m.call_tools)
                m.label = f"-> {names[0]}" + (f" (+{len(reqs) - 1})" if len(reqs) > 1 else "")

        elif t == "tool.execution_start":
            pending.setdefault(d.get("toolCallId", ""), (d.get("toolName") or "?", ""))

        elif t == "tool.execution_complete":
            name, desc = pending.get(d.get("toolCallId", ""), ("?", ""))
            result = d.get("result") or {}
            content = result.get("content")
            if content is None and d.get("error"):
                err = d["error"]
                content = err.get("message") if isinstance(err, dict) else str(err)
            content = content if isinstance(content, str) or content is None else json.dumps(content)
            if d.get("isUserRequested"):
                # ! commands end up in the context as a user message, not as a tool result.
                m = Message(role="user", label="<user_shell_command_output>")
                m.add("user", count(content))
                s.append(m)
            else:
                m = Message(role="tool", tool=name, label=desc or content or "", alt_label=content or "")
                m.add("toolresult", count(content))
                for b in result.get("binaryResultsForLlm") or []:
                    if b.get("type") == "image":
                        s.images += 1
                        m.add("toolresult", image_tokens(assets.get(b.get("assetId"))))
                s.append(m)
                current_assistant = None

        elif t == "session.compaction_start":
            compaction_mark = len(s.messages)
            s.tooldefs_reported = d.get("toolDefinitionsTokens") or s.tooldefs_reported

        elif t == "session.compaction_complete":
            if d.get("success") is False:
                compaction_mark = None
            else:
                s.compactions += 1
                limit = compaction_mark if compaction_mark is not None else len(s.messages)
                removed = limit
                pre, gone = d.get("preCompactionMessagesLength"), d.get("messagesRemoved")
                if pre and gone is not None and pre > 1:
                    # The CLI splits its message list differently from ours; carry over the share that is kept.
                    removed = limit - round(limit * max(0, pre - 1 - gone) / (pre - 1))
                for m in s.messages[:removed]:
                    m.end = s.model_calls
                summary = Message(role="user", label=d.get("summaryContent") or "")
                summary.add("summary", count(d.get("summaryContent")))
                summary.start = s.model_calls
                s.history.append(summary)
                s.messages = [summary] + s.messages[removed:]
                compaction_mark = None
                current_assistant = None

        elif t == "session.shutdown":
            s.tooldefs_reported = d.get("toolDefinitionsTokens") or s.tooldefs_reported

        if mark_id and e.get("id") == mark_id:
            s.mark = list(s.messages)
            s.mark_system = s.system_tokens

    return s


# ---------------------------------------------------------------- matching against /context

def norm(label: str) -> str:
    label = " ".join((label or "").split())
    for prefix in ("user: ", "asst: ", "tool: "):
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    return label.rstrip("…").strip()


def same_label(ctx_label: str, ours: str) -> bool:
    a, b = norm(ctx_label), norm(ours)
    if not a or not b:
        return a == b
    if a.startswith("-> ") or b.startswith("-> "):
        return a == b
    n = min(len(a), len(b), 40)
    return a[:n] == b[:n]


def classify(role: str, label: str) -> tuple[str, str | None]:
    """Category for a /context message that is not in events.jsonl."""
    text = norm(label)
    if role == "tool":
        return "toolresult", "?"
    if role == "assistant":
        if text.startswith("-> "):
            return "toolcall", text[3:].split(" ")[0]
        return "output", None
    if text.startswith(("<skill-context", "skill:")) or "Custom instructions" in text[:60]:
        return "skills", None
    if text.startswith("<user_shell_command_output>"):
        return "user", None
    if text.startswith("<overview>") or "summary" in text[:40].lower():
        return "summary", None
    if text.startswith("<"):
        return "reminders", None
    return "user", None


@dataclass
class Calibration:
    system: float = 1.0
    custom_share: float = 0.0
    tooldefs: int = 0
    mcptools: int = 0
    matched: dict[str, tuple[int, int]] = field(default_factory=dict)  # role -> (matched, total in /context)
    phantom_tokens: int = 0


def calibrate(s: Session, snap: dict | None) -> Calibration:
    if not snap or s.mark is None:
        return Calibration(tooldefs=s.tooldefs_reported)
    ca = snap["contextAttribution"]
    cat = ca.get("categories") or {}
    sp, ci = cat.get("systemPrompt", 0), cat.get("customInstructions", 0)
    cal = Calibration(
        system=(sp + ci) / s.mark_system if s.mark_system else 1.0,
        custom_share=ci / (sp + ci) if sp + ci else 0.0,
        tooldefs=cat.get("systemTools", 0) + cat.get("mcpTools", 0),
        mcptools=cat.get("mcpTools", 0),
    )
    ctx = sorted(snap.get("heaviestMessages") or [], key=lambda m: int(str(m.get("id", "-0")).rsplit("-", 1)[-1] or 0))
    if not ctx:
        return cal

    mark_set = set(map(id, s.mark))
    role_sums: dict[str, list[float]] = {}
    phantoms: list[Message] = []
    for role in ("user", "assistant", "tool"):
        ours = [m for m in s.mark if m.role == role]
        theirs = [c for c in ctx if c.get("role") == role]
        pairs: list[int | None] = [None] * len(theirs)
        j = 0
        for i, c in enumerate(theirs):
            label = c.get("label", "")
            hit = next((k for k in range(j, min(j + 8, len(ours)))
                        if same_label(label, ours[k].label) or same_label(label, ours[k].alt_label)), None)
            if hit is not None:
                pairs[i], j = hit, hit + 1
        # Fill gaps: unmatched messages between two confident matches are paired in order.
        prev_i, prev_h = -1, -1
        for ai, ah in [(i, h) for i, h in enumerate(pairs) if h is not None] + [(len(theirs), len(ours))]:
            for i, h in zip(range(prev_i + 1, ai), range(prev_h + 1, ah)):
                pairs[i] = h
            prev_i, prev_h = ai, ah

        used, matched, prev_start = set(), 0, 0
        for c, hit in zip(theirs, pairs):
            tokens = c.get("tokens", 0)
            if hit is None:
                cat_key, tool = classify(role, c.get("label", ""))
                p = Message(role=role, label=c.get("label", ""), tool=tool if role == "tool" else None, start=prev_start)
                p.add(cat_key, tokens)
                if role == "assistant" and tool:
                    p.call_tools[tool] = tokens
                phantoms.append(p)
                cal.phantom_tokens += tokens
                continue
            m = ours[hit]
            used.add(id(m))
            matched += 1
            prev_start = m.start
            if m.total <= 0:
                cat_key, _ = classify(role, c.get("label", ""))
                m.add(cat_key, 1)
            m.factor = tokens / m.total
            sums = role_sums.setdefault(role, [0.0, 0.0])
            sums[0] += tokens
            sums[1] += m.total
        missing = 0
        for m in ours:
            if id(m) not in used:
                m.in_context = False  # the CLI does not have the message (any more)
                missing += 1
        if missing:
            # Unmatched /context messages are probably the same ones; don't count them twice in the sum over calls.
            for p in phantoms:
                if p.role == role:
                    p.start = s.model_calls
        cal.matched[role] = (matched, len(theirs))

    avg = {r: (v[0] / v[1] if v[1] else 1.0) for r, v in role_sums.items()}
    for m in s.history:
        if id(m) not in mark_set or not m.in_context:
            m.factor = avg.get(m.role, 1.0)  # not in the snapshot: scale like the rest of the role
    for p in phantoms:
        s.messages.append(p)
        s.history.append(p)
    return cal


# ---------------------------------------------------------------- totals

def totals(s: Session, cal: Calibration) -> tuple[dict, dict, dict, dict]:
    now: dict[str, float] = {k: 0.0 for k in ADDITIVE + ["mcptools"]}
    cum: dict[str, float] = {k: 0.0 for k in ADDITIVE + ["mcptools"]}
    tools_now: dict[str, dict[str, float]] = {}
    tools_cum: dict[str, dict[str, float]] = {}

    def split_system(target: dict, tokens: float, calls: int) -> None:
        target["system"] += tokens * cal.system * (1 - cal.custom_share)
        target["custom"] += tokens * cal.system * cal.custom_share
        target["tooldefs"] += cal.tooldefs * calls
        target["mcptools"] += cal.mcptools * calls

    def add_msg(target: dict, tools: dict, m: Message, weight: float) -> None:
        for k, v in m.parts.items():
            target[k] += v * m.factor * weight
        for name, v in m.call_tools.items():
            t = tools.setdefault(name, {"toolcall": 0.0, "toolresult": 0.0})
            t["toolcall"] += v * m.factor * weight
        if m.role == "tool":
            t = tools.setdefault(m.tool or "?", {"toolcall": 0.0, "toolresult": 0.0})
            t["toolresult"] += m.parts.get("toolresult", 0) * m.factor * weight

    split_system(now, s.system_tokens, 1)
    for m in s.messages:
        if m.in_context:
            add_msg(now, tools_now, m, 1)
    split_system(cum, s.system_cumulative, s.model_calls)
    for m in s.history:
        calls = (m.end if m.end is not None else s.model_calls) - m.start
        if calls > 0:
            add_msg(cum, tools_cum, m, calls)
    return now, cum, tools_now, tools_cum


# ---------------------------------------------------------------- output

def fmt(n: float) -> str:
    return f"{round(n):,}".replace(",", " ")


def bar(share: float, width: int) -> str:
    full = max(0.0, min(1.0, share)) * width
    s = "█" * int(full)
    rest = full - int(full)
    if rest >= 1 / 8 and len(s) < width:
        s += "▏▎▍▌▋▊▉"[min(6, int(rest * 8) - 1)]
    return s.ljust(width)


def pct(n: float, total: float) -> float:
    return n / total if total else 0.0


def age(iso: str) -> str:
    try:
        secs = (datetime.now(timezone.utc) - datetime.fromisoformat(iso.replace("Z", "+00:00"))).total_seconds()
    except ValueError:
        return "?"
    return f"{secs:.0f} s" if secs < 120 else f"{secs / 60:.0f} min"


W = 100


def table_row(label: str, n: float, tn: float, c: float, tc: float, show_bar: bool = True) -> str:
    return (
        f"{label:36} {fmt(n):>9} {pct(n, tn):6.1%}  {bar(pct(n, tn), 16) if show_bar else ' ' * 16}   "
        f"{fmt(c):>13} {pct(c, tc):6.1%}"
    )


def render(ctx: dict, notes: list[str], opts) -> str:
    now, cum = dict(ctx["now"]), dict(ctx["cumulative"])
    for d in (now, cum):
        d["tools"] = d.get("toolcall", 0) + d.get("toolresult", 0)
    tn = sum(ctx["now"].get(k, 0) for k in ADDITIVE)
    tc = sum(ctx["cumulative"].get(k, 0) for k in ADDITIVE)
    lines = [
        f"Session  {ctx['session']}" + (f"  \"{ctx['name']}\"" if ctx["name"] else ""),
        f"Model    {ctx['model'] or '?'}  ·  {ctx['model_calls']} model calls  ·  {ctx['compactions']} compactions",
        f"Source   {ctx['source']}",
        "",
        f"{'':36} {'─────────── Context now ───────────':>35}   {'─ Input, all calls ─':>24}",
        f"{'Category':36} {'tokens':>9} {'share':>6}  {'':16}   {'tokens':>13} {'share':>6}",
        "─" * W,
    ]
    for key, label in CATEGORIES:
        if key == "tooldefs" and ctx.get("tooldefs_unknown"):
            lines.append(f"{label:36} {'?':>9} {'':6}  {'':16}   {'?':>13}")
            continue
        lines.append(table_row(label, now.get(key, 0), tn, cum.get(key, 0), tc))
        for sub, sub_label in SUBROWS.get(key, []):
            if now.get(sub, 0) or cum.get(sub, 0):
                lines.append(table_row(f"  ↳ {sub_label}", now.get(sub, 0), tn, cum.get(sub, 0), tc, show_bar=False))
    lines.append("─" * W)
    lines.append(f"{'Total':36} {fmt(tn):>9} {'':6}  {'':16}   {fmt(tc):>13}")

    win = ctx.get("window")
    if win and win.get("promptTokenLimit"):
        lines.append(
            f"{'Context window':36} {fmt(tn)} / {fmt(win['promptTokenLimit'])} ({pct(tn, win['promptTokenLimit']):.0%})"
            f"  ·  free {'?' if win['freeSpace'] is None else fmt(win['freeSpace'])}  ·  buffer {fmt(win['buffer'])}  ·  compaction at {fmt(win['compactionThreshold'])}"
        )

    if opts.by_tool:
        tools_now, tools_cum = ctx["tools_now"], ctx["tools_cumulative"]
        names = sorted(set(tools_now) | set(tools_cum),
                       key=lambda n: (-sum(tools_now.get(n, {}).values()), -sum(tools_cum.get(n, {}).values())))
        lines += ["", "Tool calls incl results per tool", "─" * W]
        for n in names[: opts.top]:
            vn, vc = tools_now.get(n, {}), tools_cum.get(n, {})
            lines.append(table_row(n[:34], sum(vn.values()), tn, sum(vc.values()), tc))
            lines.append(f"{'':36}   calls {fmt(vn.get('toolcall', 0)):>7}  ·  results {fmt(vn.get('toolresult', 0)):>7}")
        if len(names) > opts.top:
            lines.append(f"  … {len(names) - opts.top} more (use --top)")

    if opts.defs and ctx.get("sources"):
        defs = sorted((e for e in ctx["sources"] if e.get("kind") == "toolDefinition" and e.get("tokens")),
                      key=lambda e: -e["tokens"])
        lines += ["", "Tool definitions per tool (/context)", "─" * W]
        for e in defs[: opts.top]:
            lines.append(f"  {e['label'][:34]:34} {fmt(e['tokens']):>9} {pct(e['tokens'], tn):6.1%}  {bar(pct(e['tokens'], tn), 16)}")
        if len(defs) > opts.top:
            lines.append(f"  … {len(defs) - opts.top} more: {fmt(sum(e['tokens'] for e in defs[opts.top:]))} tokens")
        other = [e for e in ctx["sources"] if e.get("kind") not in ("toolDefinition", "tool", "system") and e.get("tokens")]
        if other:
            lines += ["", "Other sources (/context attribution)", "─" * W]
            for e in sorted(other, key=lambda e: -e["tokens"]):
                lines.append(f"  {(e['kind'] + ': ' + e['label'])[:34]:34} {fmt(e['tokens']):>9} {pct(e['tokens'], tn):6.1%}")

    if opts.heaviest and ctx.get("heaviest"):
        lines += ["", f"Heaviest messages in the context (/context, top {opts.heaviest})", "─" * W]
        for m in ctx["heaviest"][: opts.heaviest]:
            lines.append(f"  {m.get('role', '?'):9} {m.get('label', '')[:56]:56} {fmt(m.get('tokens', 0)):>9} {pct(m.get('tokens', 0), tn):6.1%}")

    lines += ["", "Notes:"] + [f"  · {n}" for n in notes]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Show the content mix of the current Copilot CLI session's context window.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:" + __doc__.split("Examples:", 1)[1] if __doc__ else None,
    )
    p.add_argument("--by-tool", action="store_true", help="break down 'Tool calls incl results' per tool")
    p.add_argument("--defs", action="store_true", help="show tool definitions and other sources from /context")
    p.add_argument("--heaviest", type=int, nargs="?", const=10, default=0, metavar="N",
                   help="show the N heaviest messages from /context (default 10)")
    p.add_argument("--top", type=int, default=15, help="max rows in --by-tool/--defs (default 15)")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    p.add_argument("--no-reasoning", action="store_true", help="do not count visible reasoning text as context")
    p.add_argument("--estimate-only", action="store_true", help="ignore /context data and use only events.jsonl")
    p.add_argument("--chars-per-token", type=float, default=DEFAULT_CHARS_PER_TOKEN,
                   help="heuristic for text when tiktoken is missing (default 4)")
    p.add_argument("--session", help=argparse.SUPPRESS)  # for debugging against other sessions
    opts = p.parse_args(argv)

    session_dir = resolve_session(opts.session)
    if not (session_dir / "events.jsonl").exists():
        sys.exit(f"Missing {session_dir / 'events.jsonl'}")
    snap = None if opts.estimate_only else load_snapshot(session_dir)
    count = TokenCounter(opts.chars_per_token)
    s = analyse(session_dir, count, not opts.no_reasoning, snap.get("lastEventId") if snap else None)
    if snap and s.mark is None:
        snap = None  # the snapshot does not belong to any event we recognise
    stale = False
    if snap and s.compactions and len(snap.get("heaviestMessages") or []) > 1.5 * len(s.mark) + 10:
        # After background compaction, /context may still report the message list from before the compaction.
        stale = True
        snap = dict(snap, heaviestMessages=[])

    cal = calibrate(s, snap)
    now, cum, tools_now, tools_cum = totals(s, cal)
    notes = []
    if snap:
        ca = snap["contextAttribution"]
        cat = ca.get("categories") or {}
        source = f"/context via extension (snapshot {age(snap['capturedAt'])} old, after {snap.get('lastEventType')})"
        if s.events_after_mark:
            source += f" + estimate for {s.events_after_mark} later events"
        coverage = ", ".join(f"{r} {a}/{b}" for r, (a, b) in cal.matched.items())
        if stale:
            notes.append("System prompt, custom instructions and tool definitions are the CLI's own numbers (/context); "
                         f"messages are estimated ({count.method}) because /context still shows the message list from before the compaction")
        else:
            notes.append("Totals per category group are the CLI's own numbers (/context): system prompt, custom instructions, "
                         "tool definitions and tokens per message")
            notes.append(f"Messages matched against events.jsonl: {coverage}; the split within a message "
                         f"(e.g. calls vs. reasoning) is estimated ({count.method})")
        if cal.phantom_tokens:
            notes.append(f"{fmt(cal.phantom_tokens)} tokens in messages not found in events.jsonl are classified from the /context label")
        window = {"promptTokenLimit": ca.get("promptTokenLimit", 0), "compactionThreshold": ca.get("compactionThreshold", 0),
                  "freeSpace": None if stale else cat.get("freeSpace", 0), "buffer": cat.get("buffer", 0)}
        model = ca.get("modelId") or s.model
        heaviest = sorted(snap.get("heaviestMessages") or [], key=lambda m: -m.get("tokens", 0))
    else:
        source = f"estimate from events.jsonl ({count.method})"
        if not opts.estimate_only:
            notes.append("No /context data found - install the context-breakdown plugin (copilot plugin install …) and run /extensions reload or /restart to get the CLI's own numbers")
        notes.append("Tool definitions: " + ("last number reported by the CLI" if s.tooldefs_reported else "unknown (not reported yet)"))
        window, model, heaviest = None, s.model, None
    notes.append("Input, all calls = the main agent's context summed over every model call (incl. cached tokens); subagents and "
                 "helper calls are not included, and tool definitions are assumed constant")
    if not opts.no_reasoning:
        notes.append("Reasoning = visible reasoning text; encrypted reasoning/signatures are not included")
    if s.images:
        notes.append(f"{s.images} image(s) in tool results are estimated")
    if s.attachments:
        notes.append(f"{s.attachments} attachment(s) in user messages are not counted")

    rnd = lambda d: {k: round(v) for k, v in d.items()}  # noqa: E731
    ctx = {
        "session": session_dir.name,
        "name": s.name,
        "model": model,
        "model_calls": s.model_calls,
        "compactions": s.compactions,
        "source": source,
        "now": rnd(now),
        "cumulative": rnd(cum),
        "tools_now": {n: rnd(v) for n, v in tools_now.items()},
        "tools_cumulative": {n: rnd(v) for n, v in tools_cum.items()},
        "window": window,
        "tooldefs_unknown": not cal.tooldefs,
        "sources": snap["contextAttribution"].get("entries") if snap else None,
        "heaviest": heaviest,
    }
    if opts.json:
        print(json.dumps(ctx | {"notes": notes}, ensure_ascii=False, indent=2))
    else:
        print(render(ctx, notes, opts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
