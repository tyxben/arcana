# Tools

Define tools that the LLM can call during execution.

## Authoring

The `@arcana.tool` decorator and the `Tool` adapter cover most cases.

::: arcana.sdk.tool
    options:
      show_root_full_path: false

::: arcana.sdk.Tool
    options:
      show_root_full_path: false

## Contracts

The Pydantic models that flow through the Tool Gateway. These are
stable as data contracts (see [stability spec §1.4](../guide/stability.md)).

::: arcana.contracts.tool.ToolSpec
    options:
      show_root_full_path: false

::: arcana.contracts.tool.ToolCall
    options:
      show_root_full_path: false

::: arcana.contracts.tool.ToolResult
    options:
      show_root_full_path: false

::: arcana.contracts.tool.ToolError
    options:
      show_root_full_path: false

## Enums

::: arcana.contracts.tool.ToolErrorCategory
    options:
      show_root_full_path: false

::: arcana.contracts.tool.SideEffect
    options:
      show_root_full_path: false

## Constants

`ASK_USER_TOOL_NAME` is the reserved name for the built-in
ask-user clarification tool.

::: arcana.contracts.tool.ASK_USER_TOOL_NAME
    options:
      show_root_full_path: false
