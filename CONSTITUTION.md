# The Arcana Constitution

This is not a style guide. This is the architectural law of Arcana. Every line of code, every PR, every design decision must answer to this document.

---

## Chapter I: The Core Judgment

Large language models are not assembly line workers. They are reasoning engines.

The industry keeps making the same mistake: treating LLMs like unreliable employees who need a manager watching every step. This produces frameworks that are elaborate cages -- high ceremony, low capability.

Arcana starts from a different premise:

**Over-processing suppresses capability. Good constraints release it.**

An LLM forced through a rigid Thought/Action/Observation loop on a question it could answer in one shot is not being "controlled" -- it is being handicapped. An LLM given twenty tools when it needs two is not being "empowered" -- it is being distracted. An LLM retrying the same failed call three times is not being "resilient" -- it is burning money.

The right mental model is not a pipeline. It is an operating system.

A pipeline dictates: do step 1, then step 2, then step 3.
An operating system provides: here are your capabilities, here are your boundaries, go solve the problem.

**Arcana is an operating system for LLM agents.**

---

## Chapter II: The Four Prohibitions

These are things Arcana will never do. Not "tries to avoid." Never.

### 1. No Premature Structuring

We do not nail down steps before the LLM has assessed the problem.

A plan created before understanding is a liability. Fixed step sequences assume the problem is known. The LLM should decide whether it needs a plan at all, and if so, what kind.

Bad: "First create a plan, then execute each step, then verify."
Good: "Here is the goal. You have tools. You decide the approach."

### 2. No Controllability Theater

Process elegance is not result quality. A beautifully logged ten-step execution that produces a wrong answer is worse than a messy two-step execution that produces a right one.

We do not add ceremony to make dashboards look good. We do not force structured output when free-form would work better. We do not equate "I can see every step" with "every step is correct."

Observability serves debugging, not vanity.

### 3. No Context Hoarding

The LLM's context window is working memory, not a warehouse.

We do not dump all tool schemas upfront. We do not keep the entire conversation history in every call. We do not prepend encyclopedic system prompts "just in case."

Every token in the context must earn its place in the current reasoning step.

### 4. No Mechanical Retry

When a tool call fails, repeating the same call with the same arguments is not recovery. It is denial.

Error recovery is a diagnostic act. What failed? Why? What class of error is this? What should the LLM try differently? Arcana provides structured error information and lets the LLM decide the recovery strategy -- retry with different arguments, try a different tool, narrow the scope, or accept the failure and move on.

---

## Chapter III: The Seven Principles

These are the design laws of Arcana. They are not aspirational. They are mandatory.

### Principle 1: Default to Direct, Escalate to Agent

Most requests do not need an agent loop. A question with a known answer should be answered. A single-tool task should call one tool. The full agent loop -- with its budget tracking, progress detection, and multi-step reasoning -- is reserved for problems that actually require it.

The fastest path that produces a correct result is the right path.

### Principle 2: Context as Working Set, Not Archive

Context is organized around what the LLM needs right now, not what might be useful later.

Four layers, strict discipline:
- **Identity**: Fixed. Who you are, what your boundaries are. Always present.
- **Task**: The current goal and its constraints. Present for the duration.
- **Working**: What's needed for this specific reasoning step. Changes every step.
- **External**: Everything else. Retrieved on demand, never preloaded.

Only the Working layer is actively managed per step. Tool schemas, memory, conversation history -- all External until needed.

### Principle 3: Tools as Capabilities, Not Interfaces

A tool is not an API wrapper. A tool is a capability the LLM can reason about.

Every tool must declare not just its schema, but its affordances: when to use it, what to expect from it, what failure means. The LLM should be able to look at a tool and understand not just how to call it, but whether to call it.

### Principle 4: Allow Strategy Leaps

The LLM does not owe us a step-by-step trace of its reasoning.

If the LLM can solve a three-step problem in one step, let it. If the LLM realizes mid-plan that the plan is wrong, let it pivot. If the LLM determines the answer is "I cannot do this," let it say so without forcing it through N more steps.

Policy asks "what do you want to do next?" It does not say "here is what you must do next."

### Principle 5: Structured, Actionable Feedback

When something goes wrong, the LLM receives:
- What failed (specific tool, specific step)
- What kind of failure (validation, timeout, permission, logic)
- What to prioritize next (not just "try again")

Feedback is never a raw error string. It is a diagnostic brief the LLM can reason about.

### Principle 6: Runtime as OS, Not Form Engine

The runtime provides services. It does not impose workflows.

Runtime services: budget enforcement, trace recording, tool dispatch, capability registry, working set management, error diagnostics.

These services are always available. None of them dictate the agent's strategy. The LLM calls on them as needed, like a program calls on an operating system.

### Principle 7: Judge by Outcomes, Not by Process

The measure of an agent is whether it achieved the goal, not whether it followed the prescribed steps.

Metrics that matter:
- Did it succeed?
- How much did it cost?
- Did the user get what they asked for?
- When it failed, did it eventually recover?

Metrics that do not matter:
- Did it follow the ReAct format?
- Did it use all planned steps?
- Did it produce pretty logs?

---

## Chapter IV: The Division of Responsibility

### The Framework Is Responsible For:

- **Providing capabilities**: tools, models, memory, retrieval
- **Enforcing boundaries**: budgets, permissions, safety rails
- **Recording execution**: traces, metrics, diagnostics
- **Organizing context**: working set management, compression, retrieval
- **Classifying errors**: structured diagnostics, recovery options

### The LLM Is Responsible For:

- **Understanding the goal**: what the user actually wants
- **Forming strategy**: how to approach the problem
- **Making decisions**: which tools, which order, when to stop
- **Adapting dynamically**: pivoting when things change or fail
- **Judging completion**: knowing when the goal is met

### The Inviolable Rule:

These responsibilities never reverse.

The framework never decides strategy. The LLM never enforces budgets. The framework never tells the LLM "you must use tool X next." The LLM never decides its own token limit.

When you find yourself writing code where the framework makes a judgment call that belongs to the LLM, stop. Refactor. That is a constitutional violation.

---

## Chapter V: The Contributor Compact

### For Every Pull Request

Before submitting, answer these:

1. **Direct by default?** Does this feature honor the fast path, or does it force everything through the agent loop?
2. **Working set discipline?** Does this add to the context only what's needed, or does it hoard?
3. **Capability, not interface?** If this adds a tool, can the LLM reason about when to use it?
4. **Strategy freedom?** Does this constrain how the LLM solves problems, or does it expand what problems the LLM can solve?
5. **Actionable feedback?** If this can fail, does the failure produce something the LLM can act on?
6. **OS, not workflow?** Is this a service the LLM can call, or a step the LLM is forced through?
7. **Outcome-oriented?** Does this improve result quality, or just process visibility?

### The Fundamental Question

Every new feature must answer:

**"Is this helping the LLM or constraining it?"**

If the answer is "constraining it for safety" -- that is valid. Safety is a framework responsibility.

If the answer is "constraining it for predictability" -- that is suspect. Predictability of process is not predictability of outcome.

If the answer is "constraining it because we don't trust it" -- that is a violation. Build better guardrails, not tighter cages.

---

*This constitution is a living document. It can be amended, but amendments require the same rigor as the original: a clear argument for why the change serves the LLM's capability rather than our comfort.*
