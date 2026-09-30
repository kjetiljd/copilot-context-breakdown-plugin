// Writes the /context data for the current session to <session-state>/<id>/context-breakdown.json,
// so that breakdown.py (the context-breakdown skill) can use the CLI's own token counts.
// Deliberately registers no tools, so the extension does not change the context it measures.
import { joinSession } from "@github/copilot-sdk/extension";
import { mkdir, rename, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

const session = await joinSession({});

const stateDir = session.workspacePath
    ?? join(process.env.COPILOT_HOME ?? join(homedir(), ".copilot"), "session-state", session.sessionId);
const target = join(stateDir, "context-breakdown.json");

let lastEventId = null;
let lastEventType = null;
let running = false;
let pending = false;
let modelsPromise = null;

// getContextAttribution() falls back to a default 128k limit with no output reserve, so look up the
// current model's limits for its context tier and ask for the same numbers /context shows.
async function contextWindow() {
    const current = await session.rpc.model.getCurrent().catch(() => null);
    if (!current?.modelId) return null;
    modelsPromise ??= session.connection.sendRequest("models.list", {}).catch(() => {
        modelsPromise = null;
        return null;
    });
    const model = (await modelsPromise)?.models?.find((m) => m.id === current.modelId);
    if (!model) return null;
    const tier = current.contextTier ?? "default";
    const prices = model.billing?.tokenPrices;
    const promptTokenLimit = (tier === "long_context" ? prices?.longContext?.maxPromptTokens : prices?.maxPromptTokens)
        ?? model.capabilities?.limits?.max_prompt_tokens ?? 0;
    const outputTokenLimit = model.capabilities?.limits?.max_output_tokens ?? 0;
    if (!promptTokenLimit) return null;
    const info = await session.rpc.metadata.contextInfo({
        promptTokenLimit,
        outputTokenLimit,
        selectedModel: current.modelId,
    });
    return info?.contextInfo ? { modelId: current.modelId, contextTier: tier, ...info.contextInfo } : null;
}

async function snapshot(reason) {
    if (running) {
        pending = true;
        return;
    }
    running = true;
    try {
        const eventId = lastEventId;
        const eventType = lastEventType;
        const [attribution, heaviest, window] = await Promise.all([
            session.rpc.metadata.getContextAttribution(),
            session.rpc.metadata.getContextHeaviestMessages({ limit: 100000 }).catch(() => null),
            contextWindow().catch(() => null),
        ]);
        const data = {
            version: 1,
            sessionId: session.sessionId,
            capturedAt: new Date().toISOString(),
            reason,
            lastEventId: eventId,
            lastEventType: eventType,
            contextAttribution: attribution?.contextAttribution ?? null,
            heaviestMessages: heaviest?.messages ?? null,
            contextWindow: window,
        };
        await mkdir(stateDir, { recursive: true });
        const tmp = `${target}.${process.pid}.tmp`;
        await writeFile(tmp, JSON.stringify(data, null, 2));
        await rename(tmp, target);
    } catch (err) {
        await session.log(`context-breakdown: could not fetch /context data: ${err?.message ?? err}`, {
            level: "warning",
            ephemeral: true,
        }).catch(() => {});
    } finally {
        running = false;
        if (pending) {
            pending = false;
            void snapshot("coalesced");
        }
    }
}

const TRIGGERS = new Set([
    "assistant.message",
    "session.idle",
    "session.compaction_complete",
    "session.model_change",
    "session.context_cleared",
]);

session.on((event) => {
    if (event.data?.parentToolCallId) return; // subagents have their own context
    if (!event.ephemeral) {
        // Only persisted events are in events.jsonl and can be used as a marker.
        lastEventId = event.id ?? lastEventId;
        lastEventType = event.type ?? lastEventType;
    }
    if (TRIGGERS.has(event.type)) void snapshot(event.type);
});

void snapshot("startup");
