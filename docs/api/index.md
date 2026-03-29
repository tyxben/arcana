# API Reference

Arcana's public API is designed around a single entry point (`arcana.run()`) for simple tasks and `Runtime` for production use.

## Quick Reference

| Entry Point | Use Case |
|---|---|
| `arcana.run()` | One-shot tasks, scripts, prototyping |
| `Runtime.run()` | Production single-shot with shared resources |
| `Runtime.chat()` | Multi-turn conversations |
| `Runtime.chain()` | Sequential pipelines |
| `Runtime.team()` | Multi-agent collaboration |
| `Runtime.stream()` | SSE streaming |

## Modules

- [SDK (arcana.run)](sdk.md) -- Zero-config entry point
- [Runtime](runtime.md) -- Production runtime container
- [ChatSession](chat.md) -- Multi-turn conversation
- [Budget](budget.md) -- Cost and token control
- [Tools](tools.md) -- Tool definition and adapters
- [Contracts](contracts.md) -- Data models (Pydantic schemas)
