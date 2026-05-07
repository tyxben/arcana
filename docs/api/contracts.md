# Contracts

All data flowing between layers is a Pydantic model. The names below
are part of the v1.0.0 stable contract surface
([stability spec §1.4](../guide/stability.md)) — field rename = major
bump, field addition with default = minor bump.

## Run / Pipeline / Batch

::: arcana.runtime_core.RunResult
    options:
      show_root_full_path: false

::: arcana.runtime_core.ChainStep
    options:
      show_root_full_path: false

::: arcana.runtime_core.ChainResult
    options:
      show_root_full_path: false

::: arcana.runtime_core.BatchResult
    options:
      show_root_full_path: false

## Configuration

::: arcana.runtime_core.AgentConfig
    options:
      show_root_full_path: false

::: arcana.runtime_core.RuntimeConfig
    options:
      show_root_full_path: false

## Turn (`arcana.contracts.turn`)

The V2 separation principle: facts are what the LLM said, assessment
is what the runtime concluded.

::: arcana.contracts.turn.TurnFacts
    options:
      show_root_full_path: false

::: arcana.contracts.turn.TurnAssessment
    options:
      show_root_full_path: false

## LLM (`arcana.contracts.llm`)

::: arcana.contracts.llm.Message
    options:
      show_root_full_path: false

::: arcana.contracts.llm.MessageRole
    options:
      show_root_full_path: false

::: arcana.contracts.llm.LLMRequest
    options:
      show_root_full_path: false

::: arcana.contracts.llm.LLMResponse
    options:
      show_root_full_path: false

::: arcana.contracts.llm.ContentBlock
    options:
      show_root_full_path: false

::: arcana.contracts.llm.ModelConfig
    options:
      show_root_full_path: false

## Context (`arcana.contracts.context`)

::: arcana.contracts.context.ContextBlock
    options:
      show_root_full_path: false

::: arcana.contracts.context.ContextDecision
    options:
      show_root_full_path: false

::: arcana.contracts.context.MessageDecision
    options:
      show_root_full_path: false

::: arcana.contracts.context.ContextReport
    options:
      show_root_full_path: false

::: arcana.contracts.context.ContextStrategy
    options:
      show_root_full_path: false

::: arcana.contracts.context.ContextLayer
    options:
      show_root_full_path: false

::: arcana.contracts.context.TokenBudget
    options:
      show_root_full_path: false

::: arcana.contracts.context.WorkingSet
    options:
      show_root_full_path: false

::: arcana.contracts.context.StepContext
    options:
      show_root_full_path: false

## Diagnosis (`arcana.contracts.diagnosis`)

::: arcana.contracts.diagnosis.ErrorDiagnosis
    options:
      show_root_full_path: false

::: arcana.contracts.diagnosis.ErrorCategory
    options:
      show_root_full_path: false

::: arcana.contracts.diagnosis.ErrorLayer
    options:
      show_root_full_path: false

::: arcana.contracts.diagnosis.RecoveryStrategy
    options:
      show_root_full_path: false

## Streaming (`arcana.contracts.streaming`)

::: arcana.contracts.streaming.StreamEvent
    options:
      show_root_full_path: false

::: arcana.contracts.streaming.StreamEventType
    options:
      show_root_full_path: false

## Cognitive (`arcana.contracts.cognitive`)

The opt-in cognitive primitives (`recall`, `pin`, `unpin`).

::: arcana.contracts.cognitive.RecallRequest
    options:
      show_root_full_path: false

::: arcana.contracts.cognitive.RecallResult
    options:
      show_root_full_path: false

::: arcana.contracts.cognitive.PinRequest
    options:
      show_root_full_path: false

::: arcana.contracts.cognitive.PinResult
    options:
      show_root_full_path: false

::: arcana.contracts.cognitive.UnpinRequest
    options:
      show_root_full_path: false

::: arcana.contracts.cognitive.UnpinResult
    options:
      show_root_full_path: false
