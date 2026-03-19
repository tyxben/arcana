"""Tests for WorkingSetBuilder and context contracts."""


from arcana.context.builder import WorkingSetBuilder, estimate_tokens
from arcana.contracts.context import (
    ContextBlock,
    ContextLayer,
    StepContext,
    TokenBudget,
)
from arcana.contracts.state import AgentState


class TestEstimateTokens:
    def test_english(self):
        tokens = estimate_tokens("Hello world this is a test")
        assert tokens > 0
        assert tokens < 20

    def test_cjk(self):
        tokens = estimate_tokens("你好世界")
        # CJK should count roughly 1 token per 2 chars
        assert tokens > 0

    def test_empty(self):
        tokens = estimate_tokens("")
        assert tokens >= 1  # minimum 1


class TestTokenBudget:
    def test_defaults(self):
        b = TokenBudget()
        assert b.total_window == 128000
        assert b.working_budget > 0

    def test_working_budget_calculation(self):
        b = TokenBudget(
            total_window=10000, identity_tokens=200, task_tokens=300, response_reserve=4096
        )
        assert b.working_budget == 10000 - 200 - 300 - 4096


class TestContextBlock:
    def test_basic(self):
        block = ContextBlock(
            layer=ContextLayer.WORKING,
            key="test",
            content="hello",
            token_count=2,
        )
        assert block.priority == 0.5
        assert block.compressible


class TestStepContext:
    def test_defaults(self):
        ctx = StepContext()
        assert not ctx.needs_tools
        assert not ctx.needs_memory

    def test_with_error(self):
        ctx = StepContext(
            previous_error={"category": "tool_mismatch"},
            needs_tools=True,
        )
        assert ctx.previous_error is not None


class TestWorkingSetBuilder:
    def test_build_basic(self):
        builder = WorkingSetBuilder(identity="You are a helpful assistant")
        state = AgentState(run_id="test", goal="Do something")
        ctx = StepContext()

        ws = builder.build(state, ctx)
        assert ws.identity.content == "You are a helpful assistant"
        assert "Do something" in ws.task.content
        assert ws.total_tokens > 0

    def test_build_with_tools(self):
        builder = WorkingSetBuilder(identity="Assistant")
        state = AgentState(run_id="test", goal="Search")
        ctx = StepContext(needs_tools=True)

        ws = builder.build(state, ctx, tool_descriptions="web_search: search the web")
        # Should include tool block
        tool_blocks = [b for b in ws.working_blocks if b.key == "tools"]
        assert len(tool_blocks) == 1

    def test_build_with_error(self):
        builder = WorkingSetBuilder(identity="Assistant")
        state = AgentState(run_id="test", goal="Fix")
        ctx = StepContext(previous_error={"recovery_prompt": "Try a different tool"})

        ws = builder.build(state, ctx)
        error_blocks = [b for b in ws.working_blocks if b.key == "error_context"]
        assert len(error_blocks) == 1

    def test_priority_ordering(self):
        builder = WorkingSetBuilder(
            identity="A",
            token_budget=TokenBudget(total_window=1000, response_reserve=100),
        )
        state = AgentState(run_id="test", goal="Test")
        ctx = StepContext(
            needs_tools=True,
            needs_memory=True,
            previous_error={"recovery_prompt": "error context"},
        )

        ws = builder.build(
            state,
            ctx,
            tool_descriptions="tools here",
            memory_results="memory here",
        )
        # Error context (0.95) should come before tools (0.8) and memory (0.5)
        if len(ws.working_blocks) >= 2:
            assert ws.working_blocks[0].priority >= ws.working_blocks[1].priority

    def test_to_messages(self):
        builder = WorkingSetBuilder(identity="Assistant")
        state = AgentState(run_id="test", goal="Hello")
        ctx = StepContext()

        ws = builder.build(state, ctx)
        messages = builder.to_messages(ws)
        assert len(messages) >= 1
        assert messages[0].role.value == "system"

    def test_drops_when_over_budget(self):
        builder = WorkingSetBuilder(
            identity="A",
            token_budget=TokenBudget(total_window=500, response_reserve=100),
        )
        state = AgentState(run_id="test", goal="Test")
        ctx = StepContext(needs_tools=True, needs_memory=True)

        ws = builder.build(
            state,
            ctx,
            tool_descriptions="x" * 2000,  # Very long
            memory_results="y" * 2000,  # Very long
        )
        # Should have dropped something
        assert len(ws.dropped_keys) > 0 or ws.total_tokens <= 500
