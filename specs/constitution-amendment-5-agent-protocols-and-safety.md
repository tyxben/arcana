# Constitution Amendment 5: Agent Protocols and Safety Boundaries

**Status**: accepted into Constitution v3.6

**Date**: 2026-05-27

**Amends**: Principle 3, Principle 7, Principle 8, Chapter IV, Chapter V

## Trigger

The 2025-2026 agent ecosystem moved in three directions at once:

1. Tool and data access is consolidating around open capability transports such
   as MCP and provider-hosted connectors.
2. Multi-agent systems are moving toward interoperable agent-to-agent protocols
   such as A2A, with discovery, identity, version negotiation, and cross-vendor
   message exchange becoming production concerns.
3. Security and reliability guidance increasingly treats prompt injection,
   remote tool authority, approval, traceability, evals, and guardrails as
   baseline requirements for production agents.

These trends are compatible with Arcana only if they are interpreted through the
OS model. Protocols provide transport and capability discovery. They do not
decide strategy. Guardrails provide boundaries. They do not plan. Evals provide
release evidence. They do not become runtime supervisors.

## Why this belongs in the Constitution

Arcana already has the right primitives: `ToolSpec`, `ToolErrorCategory`,
`SideEffect`, `ExecutionChannel`, trace events, working-set provenance, and
user-controlled `AgentPool` orchestration. The risk is not technical mismatch.
The risk is accidentally importing someone else's orchestration assumptions
while adding compatibility.

Examples of violations this amendment prevents:

- Treating an MCP server as safe because it is "standard" or popular.
- Importing remote tool schemas without side-effect class, approval policy, or
  provenance.
- Treating A2A support as permission to add a built-in planner/router that
  decides which remote agent speaks next.
- Letting a guardrail silently rewrite the user's goal or insert a review
  workflow.
- Treating benchmark success as proof that no runtime boundary is needed.

## Design Law

### 1. Protocols are capability transports

MCP, connectors, A2A, browser control, computer use, and future protocols enter
Arcana through contracts, not through special trust. Every imported capability
must be normalized into a runtime-visible shape:

- identity and origin
- declared affordance
- authority and authentication context
- side-effect class
- approval requirements
- provenance of tool outputs and remote-agent messages
- structured transport and protocol errors

If a protocol cannot provide those fields directly, the adapter must either
supply conservative defaults or refuse to expose the capability.

### 2. Interoperability is not orchestration

An external agent protocol can standardize discovery, identity, task envelopes,
streaming, and version negotiation. Arcana may implement those mechanics.

It must not inherit a default topology from the protocol. The user's code still
drives who speaks, when, and under what supervision policy. Protocol adapters
return addressable peers and structured failures; they do not decide strategy.

### 3. Guardrails are boundaries

Guardrails can block unsafe actions, require confirmation, redact unsafe data, or
return structured diagnostic feedback. They are constitutional when they enforce
framework responsibilities: budget, permission, safety rails, policy, and
provenance.

They are unconstitutional when they become hidden workflow managers: choosing
the next tool, forcing a plan, silently replacing the goal, or converting every
task into a review pipeline.

### 4. Evals are release evidence

Regression tests, red-team suites, alignment probes, and tool-use benchmarks are
mandatory evidence for changes that increase autonomy, remote reach, authority,
or long-running execution. Their job is to catch failure modes before release
and to make risk visible.

They do not replace runtime boundaries. Passing an eval does not authorize
silent capability escalation, prompt injection exposure, or weak provenance.

## Implementation Consequences

Near-term framework work should prefer these shapes:

- `ProtocolGateway` / adapter layer that converts MCP/A2A-style capabilities into
  existing Arcana contracts.
- Capability records with `origin`, `authority`, `side_effect`, `requires_approval`,
  and `provenance` fields before any remote tool reaches an LLM request.
- Remote-agent peers represented as addressable endpoints behind `Channel` /
  `ExecutionChannel`, not as framework-owned supervisors.
- Tool-call guardrails at the boundary of `ToolGateway` / `ExecutionChannel`,
  because remote tool calls are the highest-risk point.
- Trace events that capture protocol discovery, imported capability hashes,
  approval decisions, remote failures, and guardrail blocks.
- Focused tests for prompt-injected tool output, untrusted remote capability
  discovery, approval denial, and protocol transport failure.

## Constitutional Check

| Rule | Alignment |
|------|-----------|
| No Premature Structuring | Protocol adapters do not impose task plans. |
| No Controllability Theater | Trace and guardrails exist for debugging and safety, not dashboards. |
| No Context Hoarding | Remote catalogs are imported lazily and scoped. |
| No Mechanical Retry | Protocol failures surface as structured diagnostics. |
| Principle 3 | Remote tools are capabilities with declared affordances. |
| Principle 7 | Evals judge outcomes and risk evidence, not process shape. |
| Principle 8 | Interoperability preserves agent autonomy and user orchestration. |

## Sources Consulted

- OpenAI, "New tools for building agents" (2025-03-11): Responses API,
  built-in tools, Agents SDK, guardrails, tracing.
- OpenAI API docs, "MCP and Connectors": remote MCP servers, approvals, trust
  warnings, and protocol errors.
- Model Context Protocol specification 2025-06-18: transport-level
  authorization for HTTP transports.
- Linux Foundation, "Linux Foundation Launches the Agent2Agent Protocol Project"
  (2025-06-23): secure agent-to-agent communication, agent discovery,
  cross-platform interoperability, and vendor-neutral governance.
- OWASP MCP Top 10: prompt injection via contextual payloads, authentication and
  authorization risks.
- NIST AI RMF and NIST-AI-600-1: risk management profile for generative AI.
- Anthropic/OpenAI cross-evaluation writeup: agentic misalignment evaluations
  and tool-heavy evaluation caveats.
