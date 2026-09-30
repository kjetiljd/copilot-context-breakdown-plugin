---
name: context-breakdown
description: Show the content mix of the current session's context window (system prompt, tool definitions, user messages, tool calls/results, model output, reasoning) now and summed over all model calls. Use for /context-breakdown or questions like "what is filling the context?".
---

# context-breakdown

Run `breakdown.py` from this skill's base directory. It finds the current
session by itself, so do not pass a session id.

```bash
python3 "<base directory>/breakdown.py" [flags]
```

Pass on any flags the user gave after `/context-breakdown`, and map plain
requests to flags:

| Request                                 | Flag                  |
| --------------------------------------- | --------------------- |
| per tool                                | `--by-tool`           |
| tool definitions, MCP, skills, sources  | `--defs`              |
| heaviest messages                       | `--heaviest [N]`      |
| machine-readable                        | `--json`              |
| ignore /context data                    | `--estimate-only`     |

Run the script once, then:

1. Show the output verbatim in a `text` code block. Tool output is collapsed
   in the CLI, so the user will not see it otherwise.
2. Add at most three sentences that point out what dominates, now and summed
   over all calls. Do not recompute or reinterpret the numbers.
3. If the notes say `No /context data found`, the numbers are estimates.
   Explain that the `context-breakdown` extension from this plugin writes the
   `/context` snapshot. It must be loaded (`/extensions reload` or
   `/restart`; extensions may need `/experimental` on), and it writes the
   first snapshot after the next assistant message. Then run the skill again.

Do not run any other tools. Loading this skill and printing the output adds
a few thousand tokens to the context you are measuring.
