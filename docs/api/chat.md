# ChatSession

Multi-turn conversation with persistent history and shared budget.
Returned by `Runtime.chat()` and used as an async context manager.

The methods and properties listed below are part of the v1.0.0 stable
surface ([stability spec §1.3](../guide/stability.md)).

## `ChatSession`

::: arcana.runtime_core.ChatSession
    options:
      show_root_full_path: false
      members:
        - send
        - stream
        - history
        - max_history
        - seed_history
        - total_cost_usd
        - total_tokens
        - turn_count

## `ChatResponse`

Single-turn response yielded by `ChatSession.send()`.

::: arcana.runtime_core.ChatResponse
    options:
      show_root_full_path: false
