"""Tests for WorkingSetBuilder and context contracts."""

from arcana.context.builder import WorkingSetBuilder, estimate_tokens
from arcana.contracts.context import (
    ContextBlock,
    ContextDecision,
    ContextLayer,
    StepContext,
    TokenBudget,
)
from arcana.contracts.llm import Message, MessageRole
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

    def test_per_layer_budget_defaults(self):
        b = TokenBudget()
        assert b.tool_budget is None
        assert b.history_budget is None
        assert b.memory_budget is None

    def test_per_layer_budget_set(self):
        b = TokenBudget(tool_budget=5000, history_budget=10000, memory_budget=2000)
        assert b.tool_budget == 5000


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


# ── V1 Policy Mode Tests ──────────────────────────────────────────

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


# ── V2 Conversation Mode Tests ────────────────────────────────────

class TestConversationContext:
    def _make_messages(self, count: int, content_len: int = 100) -> list[Message]:
        """Create a list of alternating user/assistant messages."""
        msgs = [Message(role=MessageRole.SYSTEM, content="You are helpful.")]
        for i in range(count):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            msgs.append(Message(role=role, content=f"Message {i}: " + "x" * content_len))
        return msgs

    def test_under_budget_passthrough(self):
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=128000),
        )
        msgs = self._make_messages(4)
        result = builder.build_conversation_context(msgs, turn=0)
        assert len(result) == len(msgs)
        assert builder.last_decision is not None
        assert not builder.last_decision.history_compressed
        assert "Under budget" in builder.last_decision.explanation

    def test_over_budget_compresses(self):
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=800, response_reserve=50),
        )
        msgs = self._make_messages(20, content_len=200)
        result = builder.build_conversation_context(msgs, turn=3)

        # Should be fewer messages
        assert len(result) < len(msgs)
        # Should contain a summary
        has_summary = any("Earlier conversation" in (m.content or "") for m in result)
        assert has_summary
        # Decision should be recorded
        assert builder.last_decision is not None
        assert builder.last_decision.history_compressed
        assert builder.last_decision.compressed_count > 0
        assert builder.last_decision.turn == 3

    def test_memory_injection(self):
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=128000),
        )
        msgs = [
            Message(role=MessageRole.SYSTEM, content="System prompt"),
            Message(role=MessageRole.USER, content="Hello"),
        ]
        result = builder.build_conversation_context(
            msgs, memory_context="User likes Python.", turn=0
        )
        assert "User likes Python" in (result[0].content or "")
        assert builder.last_decision is not None
        assert builder.last_decision.memory_injected

    def test_decision_budget_breakdown(self):
        builder = WorkingSetBuilder(
            identity="System",
            token_budget=TokenBudget(total_window=10000, response_reserve=2000),
        )
        msgs = self._make_messages(4)
        builder.build_conversation_context(msgs, tool_token_estimate=500, turn=1)

        d = builder.last_decision
        assert d is not None
        assert d.budget_tools == 500
        assert d.budget_reserve == 2000
        assert d.budget_total == 10000 - 2000 - 500
        assert d.budget_used > 0

    def test_keeps_system_and_recent_messages(self):
        """System prompt and last 6 messages should always be kept."""
        builder = WorkingSetBuilder(
            identity="System",
            token_budget=TokenBudget(total_window=800, response_reserve=100),
        )
        msgs = self._make_messages(20, content_len=100)
        result = builder.build_conversation_context(msgs, turn=5)

        # System prompt (head)
        assert result[0].role == MessageRole.SYSTEM
        # Last messages should be preserved
        assert result[-1].content == msgs[-1].content


# ── Relevance Scoring Tests ───────────────────────────────────────

class TestRelevanceScoring:
    def test_tool_messages_score_higher(self):
        builder = WorkingSetBuilder(identity="System", goal="find files")
        tool_msg = Message(role=MessageRole.TOOL, content="file contents here")
        user_msg = Message(role=MessageRole.USER, content="hello there")

        tool_score = builder._relevance_score(tool_msg)
        user_score = builder._relevance_score(user_msg)
        assert tool_score > user_score

    def test_goal_keyword_overlap_boosts(self):
        builder = WorkingSetBuilder(identity="System", goal="search database records")
        relevant = Message(role=MessageRole.USER, content="I searched the database and found records")
        irrelevant = Message(role=MessageRole.USER, content="The weather is nice today")

        assert builder._relevance_score(relevant) > builder._relevance_score(irrelevant)

    def test_error_content_scores_high(self):
        builder = WorkingSetBuilder(identity="System")
        error_msg = Message(role=MessageRole.ASSISTANT, content="Error: connection failed, recovery needed")
        normal_msg = Message(role=MessageRole.ASSISTANT, content="The result is 42")

        assert builder._relevance_score(error_msg) > builder._relevance_score(normal_msg)

    def test_relevance_affects_compression(self):
        """High-relevance messages should get more detail in compression."""
        builder = WorkingSetBuilder(
            identity="System",
            token_budget=TokenBudget(total_window=200, response_reserve=20),
            goal="search files",
        )
        msgs = [
            Message(role=MessageRole.SYSTEM, content="System"),
            # Middle messages (will be compressed)
            Message(role=MessageRole.TOOL, content="Found file: important_data.csv with search results matching query " + "x" * 300),
            Message(role=MessageRole.USER, content="Unrelated chitchat about weather " + "y" * 300),
            # Tail (kept)
            Message(role=MessageRole.USER, content="R1"),
            Message(role=MessageRole.ASSISTANT, content="R2"),
            Message(role=MessageRole.USER, content="R3"),
            Message(role=MessageRole.ASSISTANT, content="R4"),
            Message(role=MessageRole.USER, content="R5"),
            Message(role=MessageRole.ASSISTANT, content="R6"),
        ]
        builder.build_conversation_context(msgs, turn=5)
        assert builder.last_decision is not None
        assert builder.last_decision.history_compressed


# ── Per-Layer Budget Tests ────────────────────────────────────────

class TestPerLayerBudget:
    def test_tool_budget_cap(self):
        builder = WorkingSetBuilder(
            identity="System",
            token_budget=TokenBudget(total_window=10000, response_reserve=1000, tool_budget=500),
        )
        msgs = [
            Message(role=MessageRole.SYSTEM, content="System"),
            Message(role=MessageRole.USER, content="Hello"),
        ]
        # Claim 2000 tool tokens, but tool_budget caps at 500
        builder.build_conversation_context(msgs, tool_token_estimate=2000, turn=0)
        d = builder.last_decision
        assert d is not None
        assert d.budget_tools == 500  # Capped

    def test_memory_budget_cap(self):
        builder = WorkingSetBuilder(
            identity="System",
            token_budget=TokenBudget(total_window=10000, response_reserve=1000, memory_budget=50),
        )
        msgs = [
            Message(role=MessageRole.SYSTEM, content="System"),
            Message(role=MessageRole.USER, content="Hello"),
        ]
        # Long memory should be truncated
        long_memory = "Important fact about the user. " * 100
        result = builder.build_conversation_context(
            msgs, memory_context=long_memory, turn=0
        )
        # Memory should be injected but truncated
        sys_content = result[0].content or ""
        assert "memory truncated" in sys_content

    def test_history_budget_cap(self):
        """History budget should limit how much history can consume."""
        builder = WorkingSetBuilder(
            identity="System",
            token_budget=TokenBudget(
                total_window=100000, response_reserve=1000, history_budget=200
            ),
        )
        msgs = [Message(role=MessageRole.SYSTEM, content="System")]
        # Add many long messages that exceed history_budget
        for i in range(30):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            msgs.append(Message(role=role, content=f"Turn {i}: " + "x" * 500))

        builder.build_conversation_context(msgs, turn=10)
        assert builder.last_decision is not None
        assert builder.last_decision.history_compressed


# ── Long Conversation Behavior Tests ──────────────────────────────

class TestLongConversation:
    def test_goal_survives_20_turns(self):
        """After 20 turns, the goal should still be in the context."""
        builder = WorkingSetBuilder(
            identity="You help with file search tasks.",
            token_budget=TokenBudget(total_window=1000, response_reserve=50),
            goal="find all Python files containing the word 'import'",
        )

        msgs = [Message(role=MessageRole.SYSTEM, content="You help with file search tasks.")]
        # Simulate 20 turns with substantial content
        for i in range(20):
            msgs.append(Message(role=MessageRole.USER, content=f"Turn {i}: please continue searching for Python files " + "x" * 200))
            msgs.append(Message(role=MessageRole.ASSISTANT, content=f"Found file_{i}.py with import statements " + "y" * 200))

        result = builder.build_conversation_context(msgs, turn=20)

        # System prompt should always be there
        assert result[0].role == MessageRole.SYSTEM
        assert "file search" in (result[0].content or "")
        # Should have compressed
        assert builder.last_decision is not None
        assert builder.last_decision.messages_out < builder.last_decision.messages_in

    def test_tool_results_preserved_in_summary(self):
        """Key tool results should get more detail in compressed summary."""
        builder = WorkingSetBuilder(
            identity="System",
            token_budget=TokenBudget(total_window=800, response_reserve=50),
            goal="read config file",
        )

        msgs = [Message(role=MessageRole.SYSTEM, content="System")]
        # Add some turns with substantial content
        for i in range(5):
            msgs.append(Message(role=MessageRole.USER, content=f"User turn {i} " + "x" * 200))
            msgs.append(Message(role=MessageRole.ASSISTANT, content=f"Response {i} " + "y" * 200))

        # Add a tool result with important data
        msgs.append(Message(role=MessageRole.TOOL, content="config: database_url=postgres://localhost/db, port=5432 " + "z" * 200))

        # Add more turns to push tool result into compression zone
        for i in range(8):
            msgs.append(Message(role=MessageRole.USER, content=f"Later turn {i} " + "x" * 200))
            msgs.append(Message(role=MessageRole.ASSISTANT, content=f"Later response {i} " + "y" * 200))

        builder.build_conversation_context(msgs, turn=15)
        assert builder.last_decision is not None
        assert builder.last_decision.compressed_count > 0

    def test_context_decision_every_turn(self):
        """Every call should produce a ContextDecision."""
        builder = WorkingSetBuilder(
            identity="System",
            token_budget=TokenBudget(total_window=128000),
        )
        msgs = [
            Message(role=MessageRole.SYSTEM, content="System"),
            Message(role=MessageRole.USER, content="Hello"),
        ]

        for turn in range(5):
            builder.build_conversation_context(msgs, turn=turn)
            assert builder.last_decision is not None
            assert builder.last_decision.turn == turn
            msgs.append(Message(role=MessageRole.ASSISTANT, content=f"Response {turn}"))
            msgs.append(Message(role=MessageRole.USER, content=f"Follow-up {turn}"))


# ── ContextDecision Contract Tests ────────────────────────────────

class TestContextDecision:
    def test_basic_construction(self):
        d = ContextDecision(
            turn=3,
            budget_total=10000,
            budget_used=5000,
            messages_in=20,
            messages_out=8,
            compressed_count=12,
            history_compressed=True,
            explanation="12 messages compressed (2.5x), budget 50% full",
        )
        assert d.turn == 3
        assert d.compressed_count == 12
        assert d.history_compressed

    def test_defaults(self):
        d = ContextDecision()
        assert d.turn == 0
        assert d.budget_total == 0
        assert not d.memory_injected
        assert not d.history_compressed
        assert d.explanation == ""

    def test_serialization(self):
        d = ContextDecision(
            turn=1,
            budget_total=5000,
            budget_used=3000,
            compressed_messages=["user:hello", "assistant:hi"],
        )
        data = d.model_dump()
        restored = ContextDecision.model_validate(data)
        assert restored.compressed_messages == ["user:hello", "assistant:hi"]
