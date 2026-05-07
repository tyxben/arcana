# Runtime

Long-lived resource container. Create once at startup, share across an application.
Holds providers, tools, budget, and trace configuration.

The methods listed below are part of the v1.0.0 stable surface
([stability spec §1.2](../guide/stability.md)). Other public attributes
on `Runtime` (e.g. `make_llm_node`, `connect_mcp`) work today but are
**not** stability-promised.

## `Runtime`

::: arcana.runtime_core.Runtime
    options:
      show_root_full_path: false
      members:
        - __init__
        - run
        - run_batch
        - chat
        - chain
        - collaborate
        - session
        - close
        - "on"
        - "off"
        - budget_remaining_usd
        - budget_used_usd
        - tokens_used
        - tokens_remaining
