# copilot-context-breakdown-plugin

A GitHub Copilot CLI plugin that shows the content mix of the current
session's context window (system prompt, tool definitions, user messages,
tool calls incl. results, model outputs, reasoning), both now and summed over
all model calls.

## Install

```sh
copilot plugin marketplace add kjetiljd/copilot-context-breakdown-plugin   # or a local path to the repo
copilot plugin install context-breakdown@copilot-context-breakdown-plugin
```

The extension is loaded when a session starts (or after `/restart`).

## Use

Run the `/context-breakdown` skill, or run the script directly:

```sh
!python3 plugins/context-breakdown/skills/context-breakdown/breakdown.py --help
```

The plugin's extension fetches the CLI's own `/context` numbers so the script
can use them as ground truth; without it, the script falls back to estimates
from `events.jsonl`.
