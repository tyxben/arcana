# Multi-Agent Collaboration

Arcana's constitution is explicit about multi-agent: **the framework provides
coordination infrastructure, never orchestration strategy**. This guide shows
what that looks like in practice.

If you are looking for declarative agent graphs, role hierarchies, or a
built-in "who speaks next" scheduler, Arcana does not provide them — and that
is not an oversight. They belong to the LLM and the user, not to the
framework. See Principle 8 in `CONSTITUTION.md` for the full argument.

What Arcana *does* provide:

- **`AgentPool`** — named [ChatSessions](quickstart.md#chat-sessions) sharing
  one budget, one message channel, one key-value context.
- **`Channel`** — async, name-addressed message queue for when direct
  `session.send(...)` control flow is not enough.
- **`SharedContext`** — thread-safe key-value store agents can read and write.
- **Per-agent [cognitive primitives](cognitive-primitives.md)** — each pool
  member has its own `PinState`, its own recall log, its own compression
  budget.
- **Pool-aware trace replay** — every emitted event carries the originating
  agent name.

The rest is plain Python: `async for`, `if`, `await`, your own loops, your
own stop conditions.

---

## Getting started

```python
import arcana

runtime = arcana.Runtime(providers={"deepseek": "sk-..."})

async with runtime.collaborate() as pool:
    planner  = pool.add("planner",  system="You break tasks into steps.")
    executor = pool.add("executor", system="You execute one step at a time.",
                        tools=[run_shell])

    plan = await planner.send(f"Plan: {goal}")
    for step in parse_steps(plan.content):
        result = await executor.send(f"Execute: {step}")
        print(result.content)
```

`runtime.collaborate()` returns an `AgentPool` — enter it with `async with`
for automatic cleanup on exit. Inside the pool, every `pool.add(...)` creates
an independent `ChatSession` that shares the pool's `Channel`,
`SharedContext`, and `BudgetTracker` but has its own system prompt, tools,
and history.

!!! info "The pool is not a runner"
    Nothing runs until *you* call `await session.send(...)`. Two pool agents
    do not take turns unless your code makes them. This is deliberate.

---

## Four patterns

These are the most common multi-agent shapes. None of them require framework
support beyond the three primitives above — they are all plain Python.

### 1. Planner / Executor

One agent plans, another executes. Sequential, no loop.

```python
async with runtime.collaborate() as pool:
    planner  = pool.add("planner",  system="You plan.")
    executor = pool.add("executor", system="You execute.", tools=[...])

    plan = await planner.send(f"Plan a migration for {project}")
    for step in parse_steps(plan.content):
        await executor.send(f"Execute step: {step}")
```

### 2. Critic loop

A worker drafts, a critic reviews, the worker revises — capped at N rounds.

```python
async with runtime.collaborate(cognitive_primitives=["pin"]) as pool:
    worker = pool.add("worker", system="You write code.")
    critic = pool.add("critic", system="You review code strictly.")

    draft = await worker.send(f"Write: {spec}")
    for _ in range(MAX_ITERS):
        review = await critic.send(f"Review this draft:\n{draft.content}")
        if "LGTM" in review.content:
            break
        draft = await worker.send(f"Revise based on:\n{review.content}")
```

!!! tip "Why pin here?"
    The worker can `pin` the latest spec so repeated revisions do not
    accidentally compress it out of context. Each pool agent's pins are
    **private** — the critic's pins do not leak to the worker.

### 3. Specialization

Different agents, different tools. Useful when tool surface area matters.

```python
async with runtime.collaborate() as pool:
    researcher = pool.add("researcher",
                          system="You research.",
                          tools=[web_search, fetch_url])
    writer     = pool.add("writer",
                          system="You write structured notes.",
                          tools=[save_note])

    findings = await researcher.send(f"Research {topic}; cite sources.")
    await writer.send(f"Turn this into a report:\n{findings.content}")
```

### 4. Debate via Channel

When the shape is not a linear pipeline. `Channel` is an async queue; each
agent picks up messages on its own schedule.

```python
from arcana.contracts.multi_agent import ChannelMessage

async with runtime.collaborate() as pool:
    bull = pool.add("bull", system="Argue for position X.")
    bear = pool.add("bear", system="Argue against position X.")

    opening = await bull.send("State your thesis.")
    await pool.channel.send(
        ChannelMessage(sender="bull", recipient="bear",
                       content=opening.content)
    )

    for _ in range(MAX_ROUNDS):
        incoming = await pool.channel.receive("bear")
        if not incoming:
            break
        rebuttal = await bear.send(f"Rebut: {incoming[-1].content}")
        await pool.channel.send(
            ChannelMessage(sender="bear", recipient="bull",
                           content=rebuttal.content)
        )
        # ... symmetric loop continues
```

`ChannelMessage` is immutable — a single instance fans out to every
recipient and to `pool.channel.history` without any mutation hazard.

---

## Per-agent cognitive primitives

Each pool member is an independent cognitive instance. This is the feature
that meaningfully distinguishes multi-agent Arcana from orchestration-first
frameworks.

```python
async with runtime.collaborate(cognitive_primitives=["pin"]) as pool:
    a = pool.add("a")                                    # inherits ["pin"]
    b = pool.add("b", cognitive_primitives=["recall"])   # overrides
    c = pool.add("c", cognitive_primitives=[])           # explicit opt-out
```

Resolution order:

1. **Per-agent** `cognitive_primitives=[...]` (including `[]`)
2. **Pool default** passed to `runtime.collaborate(...)`
3. **Runtime default** from `RuntimeConfig.cognitive_primitives`

Every pool agent gets its **own** `PinState`, its **own** recall log, its
**own** `pin_budget_fraction` cap. Cognitive state never crosses between
pool members — see Principle 8 in `CONSTITUTION.md`.

!!! warning "Tool-name collisions raise"
    If you enable `cognitive_primitives=["pin"]` on an agent and also pass a
    user tool named `pin` (or `unpin`, which the pin primitive implicitly
    reserves), `pool.add(...)` raises `ValueError`. Silent shadowing would
    drop information either way; rename the user tool or drop the primitive
    deliberately.

---

## Shared budget

All agents in a single pool share one `BudgetTracker`:

```python
async with runtime.collaborate(budget=arcana.Budget(max_cost_usd=2.0)) as pool:
    ...
```

One agent's overspending starves the others — that is what "pool budget"
means. If you want independent budgets, create independent pools (or run the
agents outside a pool entirely).

Pin budget (`pin_budget_fraction`, default 50%) is **per agent**, not per
pool — it is computed against each agent's own context window.

---

## Shared state

`pool.shared` is a thread-safe key-value store. Any agent can read or write.
There is no access control — all agents see everything, per Principle 8.

```python
async with runtime.collaborate() as pool:
    researcher = pool.add("researcher")
    reviewer   = pool.add("reviewer")

    findings = await researcher.send(f"Research {topic}")
    pool.shared.set("findings", findings.content)

    # Reviewer reads the same value
    await reviewer.send(
        f"Review: {pool.shared.get('findings')}"
    )
```

Use `shared` for small, structured hand-offs between agents.
For unstructured back-and-forth, `Channel` is usually the better fit.

---

## Bounding channel history (v0.8.1+)

`Channel.history` retains every message ever sent. For short-lived pools
this is fine. For daemon-style pools that run indefinitely, it turns into
a slow memory leak, so `collaborate()` accepts an opt-in bound:

```python
async with runtime.collaborate(channel_history_limit=500) as pool:
    planner = pool.add("planner", system="...")
    executor = pool.add("executor", system="...")
    # pool.channel.history now retains at most the 500 most recent messages;
    # oldest entries are evicted FIFO. Delivery is unaffected.
```

- `None` (default) — unbounded history, matches pre-v0.8.1 behaviour.
- positive `int` — retain at most that many past messages.
- `0` — disable history entirely (useful when you only need live
  delivery and never introspect `channel.history`).
- negative — raises `ValueError`.

This bounds only `channel.history`. Per-agent delivery queues are driven
by your `receive()` calls; an agent that is registered but never reads
from its queue will still accumulate messages there, which is a consumer
concern rather than a retention one.

---

## Tracing pool runs

Every event emitted during a pool run carries
`metadata["source_agent"] = <pool_name>`. The `TraceEvent` schema itself is
unchanged, so v0.6.0/v0.7.0 tooling keeps working.

```bash
# Summary: which agents participated, how many events, replayable turns
arcana trace pool-replay <run_id>

# Scope output to one agent
arcana trace show <run_id> --agent planner
arcana trace replay <run_id> --agent planner --turn 3

# Cognitive-primitive events show [source_agent] in the event listing
arcana trace show <run_id> --cognitive
```

Pin state is reconstructed per agent during `trace replay` — the "active
pins at turn N" section belongs to the agent you scoped to.

---

## What you will not find in Arcana

The following features are deliberately absent. They are not on a roadmap
and will not be added.

- **Graph DSL.** No `StateGraph`-like construct for multi-agent flows.
- **Turn scheduler.** No built-in `GroupChat`. Who talks when is your code.
- **Role hierarchy.** No `supervisor` / `worker` primitives. Roles live in
  system prompts, not in framework types.
- **Auto stop conditions.** No convergence detection, no "if consensus,
  halt." Stop when your code decides to stop.
- **Cross-agent cognitive inheritance.** Agent A's pins are not visible to
  agent B, ever. If you want agent B to see A's conclusions, call
  `pool.shared.set(...)` explicitly.

If any of these are load-bearing for your use case, LangGraph / AutoGen /
CrewAI will serve you better than Arcana. The constitutional contrast is
the point, not a gap.

---

## Migration from `runtime.team()`

`runtime.team()` was deprecated in v0.8.0 and **removed in v1.0.0**. Use
`runtime.collaborate()` instead.

```python
# Old (removed in v1.0.0)
result = await runtime.team(
    "Write a blog post about X",
    agents=[AgentConfig(name="researcher", prompt="..."),
            AgentConfig(name="writer",     prompt="...")],
    mode="shared",
)

# New
async with runtime.collaborate() as pool:
    researcher = pool.add("researcher", system="...")
    writer     = pool.add("writer",     system="...")

    findings = await researcher.send("Research X")
    post     = await writer.send(f"Write a blog post using:\n{findings.content}")
```

The new shape gives you explicit control over turn order, per-agent tool
surface, and per-agent cognition — none of which `runtime.team()` exposed.

---

## Migration from `TeamOrchestrator` and `MessageBus`

`arcana.multi_agent.team.TeamOrchestrator`, `arcana.multi_agent.team.RoleConfig`,
and `arcana.multi_agent.message_bus.MessageBus` are **deprecated as of
2026-05-03** (Constitution Amendment 3, v3.4) and emit a `DeprecationWarning`
on construction. They are slated for physical removal in a v1.x minor.

These classes encoded a framework-prescribed Planner→Executor→Critic
topology via the `AgentRole` enum (`PLANNER` / `EXECUTOR` / `CRITIC`).
Amendment 3 (and Principle 8 before it) makes that shape the user's
decision, not the framework's. The replacement is plain `runtime.collaborate()`
with whatever loop you want to write.

```python
# Old (deprecated)
from arcana.multi_agent.team import TeamOrchestrator, RoleConfig
from arcana.contracts.trace import AgentRole

orchestrator = TeamOrchestrator(
    role_configs={
        AgentRole.PLANNER:  RoleConfig(role=AgentRole.PLANNER,  policy=..., reducer=...),
        AgentRole.EXECUTOR: RoleConfig(role=AgentRole.EXECUTOR, policy=..., reducer=...),
        AgentRole.CRITIC:   RoleConfig(role=AgentRole.CRITIC,   policy=..., reducer=...),
    },
    gateway=gateway,
    max_rounds=5,
)
result = await orchestrator.run(goal)

# New — your code drives the loop
async with runtime.collaborate() as pool:
    planner  = pool.add("planner",  system="You break tasks into plans.")
    executor = pool.add("executor", system="You execute one step at a time.",
                        tools=[...])
    critic   = pool.add("critic",   system="You verify execution. Reply 'pass' or feedback.")

    feedback = ""
    for _ in range(MAX_ROUNDS):
        plan_msg   = await planner.send(f"Goal: {goal}\nFeedback so far: {feedback}")
        result_msg = await executor.send(f"Execute: {plan_msg.content}")
        verdict    = await critic.send(f"Verify:\n{result_msg.content}")
        if verdict.content.lower().strip().startswith("pass"):
            break
        feedback = verdict.content
```

For role-addressed message-passing (`MessageBus`), the replacement is the
name-addressed `Channel` already in `arcana.multi_agent.channel` — same
publish/subscribe shape, but addressing is by free-form agent name rather
than by a fixed `AgentRole` enum. See the four patterns at the top of this
guide for `Channel` examples.

The `AgentRole` enum itself is **not** deprecated yet — it appears on
`TraceEvent.role`, which is part of the stable `arcana.contracts.trace`
surface. Its removal will follow once the deprecated classes above ship a
removal in a future minor; that refactor is tracked in
[`specs/constitution-amendment-3-multi-agent-os.md`](../../specs/constitution-amendment-3-multi-agent-os.md)
under "Implementation follow-up."
