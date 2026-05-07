# Budget

Cost and token control for agent execution. Constrains how much an
agent (or a session, chain, or batch) is allowed to spend.

## `Budget`

::: arcana.runtime_core.Budget
    options:
      show_root_full_path: false

## `BudgetScope`

Scoped budget carved out of a parent — used by `Runtime.budget_scope()`
and `ChainStep.budget` to limit cost on a per-step or per-segment
basis.

::: arcana.runtime_core.BudgetScope
    options:
      show_root_full_path: false
