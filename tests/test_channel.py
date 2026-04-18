"""Tests for Channel -- name-addressed async message passing."""

from __future__ import annotations

import pytest

from arcana.contracts.multi_agent import ChannelMessage, MessageType
from arcana.multi_agent.channel import Channel

# -- helpers -------------------------------------------------------------------


def _msg(
    sender: str,
    recipient: str | None = None,
    content: str = "hello",
) -> ChannelMessage:
    return ChannelMessage(
        sender=sender,
        recipient=recipient,
        content=content,
        message_type=MessageType.CHAT,
        session_id="s1",
    )


# -- tests ---------------------------------------------------------------------


class TestChannel:
    @pytest.mark.asyncio
    async def test_point_to_point_delivery(self):
        """Recipient receives the message, others do not."""
        ch = Channel()
        ch.register("alice")
        ch.register("bob")
        ch.register("carol")

        await ch.send(_msg("alice", "bob", "for bob only"))

        bob_msgs = await ch.receive("bob")
        assert len(bob_msgs) == 1
        assert bob_msgs[0].content == "for bob only"

        carol_msgs = await ch.receive("carol")
        assert carol_msgs == []

    @pytest.mark.asyncio
    async def test_broadcast_delivery(self):
        """Broadcast reaches all agents except the sender."""
        ch = Channel()
        ch.register("alice")
        ch.register("bob")
        ch.register("carol")

        await ch.send(_msg("alice", None, "hey everyone"))

        bob_msgs = await ch.receive("bob")
        carol_msgs = await ch.receive("carol")
        alice_msgs = await ch.receive("alice")

        assert len(bob_msgs) == 1
        assert len(carol_msgs) == 1
        assert alice_msgs == []

    @pytest.mark.asyncio
    async def test_receive_drains_queue(self):
        """Second receive call returns empty after drain."""
        ch = Channel()
        ch.register("bob")

        await ch.send(_msg("alice", "bob", "msg1"))
        await ch.send(_msg("alice", "bob", "msg2"))

        first = await ch.receive("bob")
        assert len(first) == 2

        second = await ch.receive("bob")
        assert second == []

    @pytest.mark.asyncio
    async def test_register_creates_queue(self):
        """register() makes the agent visible in .agents."""
        ch = Channel()
        ch.register("alpha")
        ch.register("beta")

        assert "alpha" in ch.agents
        assert "beta" in ch.agents

    @pytest.mark.asyncio
    async def test_agents_property(self):
        """agents property lists all registered names."""
        ch = Channel()
        assert ch.agents == []

        ch.register("x")
        ch.register("y")
        assert set(ch.agents) == {"x", "y"}

    @pytest.mark.asyncio
    async def test_history_tracks_all_messages(self):
        """history property contains every sent message."""
        ch = Channel()
        ch.register("a")
        ch.register("b")

        await ch.send(_msg("a", "b", "m1"))
        await ch.send(_msg("b", "a", "m2"))
        await ch.send(_msg("a", None, "m3"))

        assert len(ch.history) == 3
        assert [m.content for m in ch.history] == ["m1", "m2", "m3"]

    @pytest.mark.asyncio
    async def test_clear_empties_everything(self):
        """clear() removes all queues and history."""
        ch = Channel()
        ch.register("a")
        await ch.send(_msg("a", "a", "self-msg"))

        ch.clear()
        assert ch.agents == []
        assert ch.history == []

    @pytest.mark.asyncio
    async def test_unregistered_recipient_auto_creates_queue(self):
        """Sending to an unregistered name creates its queue on the fly."""
        ch = Channel()
        # "zeta" was never registered
        await ch.send(_msg("alpha", "zeta", "surprise"))

        assert "zeta" in ch.agents
        msgs = await ch.receive("zeta")
        assert len(msgs) == 1
        assert msgs[0].content == "surprise"


class TestChannelMessageContract:
    def test_default_message_type(self):
        """ChannelMessage defaults to CHAT."""
        msg = ChannelMessage(sender="a", content="hi")
        assert msg.message_type == MessageType.CHAT

    def test_serialization_roundtrip(self):
        """ChannelMessage survives JSON roundtrip."""
        msg = _msg("alice", "bob", "test")
        data = msg.model_dump(mode="json")
        restored = ChannelMessage.model_validate(data)
        assert restored.sender == "alice"
        assert restored.recipient == "bob"
        assert restored.content == "test"
        assert restored.message_type == MessageType.CHAT

    def test_broadcast_recipient_is_none(self):
        """Broadcast message has recipient=None."""
        msg = ChannelMessage(sender="x", content="broadcast")
        assert msg.recipient is None

    def test_frozen_rejects_field_mutation(self):
        """Bug 2 regression: ChannelMessage is immutable, so the single
        instance fanned out to broadcast recipients cannot be mutated by
        one receiver in a way that bleeds to the others."""
        from pydantic import ValidationError

        msg = ChannelMessage(sender="a", content="original")

        with pytest.raises(ValidationError):
            msg.content = "mutated"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            msg.recipient = "someone"  # type: ignore[misc]

    def test_frozen_allows_model_copy(self):
        """``model_copy(update=...)`` is the supported way to derive a
        modified ChannelMessage from a frozen original."""
        msg = ChannelMessage(sender="a", content="v1")
        derived = msg.model_copy(update={"content": "v2"})

        assert msg.content == "v1"  # original untouched
        assert derived.content == "v2"
        assert derived.sender == msg.sender  # unchanged fields preserved

    @pytest.mark.asyncio
    async def test_broadcast_shares_one_instance_safely(self):
        """Bug 2 regression at the Channel level: the same ChannelMessage
        lands in history and in every recipient queue, and because it is
        frozen, no receiver can mutate what another receiver observes."""
        ch = Channel()
        ch.register("alice")
        ch.register("bob")
        ch.register("carol")

        await ch.send(_msg("alice", None, "broadcast-me"))

        bob_msgs = await ch.receive("bob")
        carol_msgs = await ch.receive("carol")

        assert len(bob_msgs) == 1
        assert len(carol_msgs) == 1
        # Both receivers and history see identical content (same instance,
        # which is safe because the contract is immutable).
        assert bob_msgs[0].content == carol_msgs[0].content == "broadcast-me"
        assert ch.history[0].content == "broadcast-me"
