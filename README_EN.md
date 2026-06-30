# AutoReply Judge Plugin

An **AstrBot** plugin that submits group chat messages to an LLM for semantic analysis, intelligently determining whether an automatic reply should be triggered.

---

## Features

- **LLM-Powered Judgment** — Message content along with recent context is sent to an LLM to decide whether a reply is warranted.
- **Probabilistic Fallback** — When the LLM deems a reply unnecessary, a configurable probability allows occasional replies to keep interactions lively.
- **In-Group Toggle Command** — Send `/reply` in any group to enable or disable the judgment function on the fly.
- **Independent Judge Model** — Select a lightweight model dedicated to judgment to reduce costs, while the main model handles actual conversations.

---

## Installation

Place the plugin folder into `data/plugins/`, reload plugins, then enable it from the management panel.

---

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| `enabled` | Master switch | `true` |
| `judge_provider` | Model provider for judgment (dropdown; leave empty to use the conversation model) | (empty) |
| `judge_prompt` | Prompt template for the judgment LLM (variables: `{message}`, `{context}`, `{sender}`) | See below |
| `context_size` | Number of recent messages to include as context (set to `0` to disable) | `3` |
| `history_maxlen` | Max number of recent messages cached locally (affects local cache only, not the context sent to LLM) | `10` |
| `reply_chance` | Probability (0–100) of replying even when the LLM says no | `20` |

### Default Judge Prompt

```
You are a group chat bot assistant. Determine whether the following message requires a reply.

Rules:
1. The message mentions the bot's name or @'s the bot → Reply
2. The message asks a question or requests help → Reply
3. The message is interacting with the bot → Reply
4. Casual chatting unrelated to the bot → Do not reply
5. Pure emoji or meaningless messages → Do not reply

Respond strictly in JSON format (JSON only):
{"should_reply": true/false, "confidence": 0-100, "reason": "Brief reason"}

Current message:
{message}

Recent context:
{context}
```

**Available variables:** `{message}`, `{context}`, `{sender}`

---

## Commands

Send in any group chat:

```
/reply
```

Toggles the judgment function on/off for that group.  
You can also specify a state directly: `/reply true` or `/reply false`.

> 🧠 **Persistent state:** Group switch states are automatically saved to `_group_switches.json` and restored after AstrBot restart — no need to reconfigure every time.

---

## About the ERROR Log on Interception

When the plugin determines that a message does not need a reply and terminates the pipeline, AstrBot's built-in handler raises a `GeneratorExit` exception due to the generator being prematurely closed. This is standard behavior when a generator is closed normally. The log will display:

```
[ERRO] [astrbot.main:222]: Traceback (most recent call last):
File "...", line ..., in on_message
    yield event.request_llm(
GeneratorExit
[ERRO] [astrbot.main:223]: 主动回复失败:
```

This exception **does not affect functionality** — the message has been successfully intercepted and will not be sent to the group. The plugin automatically filters these logs via `logging.Filter` during initialization to keep the console output clean.

---

Built by **DeepSeek-V4** (core development) and **chengxing507** (debugging & improvements).

---

## License

MIT License © 2026 chengxing507

---

## File Structure

```
astrbot_plugin_autoreply_judge/
├── metadata.yaml
├── _conf_schema.json
├── main.py
├── README.md
├── README_EN.md
└── LICENSE

---

## Changelog

### v1.2.0 (2026-06-30)

- Fixed plugin not working in some scenarios

### v1.1.1 (2026-06-29)

- Fixed concurrency race in `_judging_groups` causing redundant LLM calls
- Fixed `/reply` state loss when multiple groups toggle simultaneously
- Fixed switch file corruption on crash (atomic write)
- Fixed token waste when `enabled` is turned off
- Fixed cache cleanup frequency being doubled
- Fixed memory leak due to missing cleanup in write path
- Fixed redundant LLM calls from cache write race (double-check)
- Fixed `bool("false")` evaluating to `True` causing interception failure
- Fixed lock leak from `yield` inside `toggle_reply()`
- Fixed race condition in `terminate()` during plugin unload
- Fixed group ID `"0"` being misidentified as non-group
- Fixed cache cleanup exception crashing message processing
- Added missing `confidence` field in probability reply logs
- Added `{sender}` variable documentation in `_conf_schema.json`

### v1.1 (2026-06-29)

- ✨ **Private Chat Filter** — Non-group messages are now ignored, preventing unintended processing
- 🛡️ **Recursion Guard** — Added `_judging` flag to prevent recursive `on_llm_request` triggers from internal LLM calls
- 🧹 **Cache Expiry** — 120s TTL with automatic cache cleanup to prevent memory leaks
- ⏱ **Timeout Protection** — 15s timeout for LLM judgment, auto-allow on timeout
- 🔒 **Enhanced JSON Parsing** — Stack-matching + trailing comma fixing for better LLM output resilience
- 💾 **Persistent Switches** — Group toggle states are auto-saved and restored across restarts

### v1.0.1

- Fixed metadata description field
- Improved log output formatting

### v1.0.0

- Initial release
- Basic LLM auto-reply judgment
- In-group `/reply` toggle command
- Probabilistic fallback mechanism
```