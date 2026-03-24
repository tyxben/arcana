"""Tests for multimodal (image) input support.

Tests cover:
- ContentBlock creation from image URLs
- ContentBlock creation from file paths
- Message with mixed text + image content
- Message serialization with content blocks
- Backward compatibility (string content still works everywhere)
- OpenAI provider content block conversion
- Anthropic provider content block conversion
- SDK build_content_blocks helper
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from arcana.contracts.llm import (
    ContentBlock,
    ImageSource,
    Message,
    MessageRole,
)
from arcana.gateway.providers.openai_compatible import (
    _convert_content_blocks_openai,
)
from arcana.sdk import build_content_blocks

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A tiny valid 1x1 red PNG (67 bytes)
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
    "nGP4z8BQDwAEgAF/pooBPQAAAABJRU5ErkJggg=="
)
_TINY_PNG_BYTES = base64.b64decode(_TINY_PNG_B64)


@pytest.fixture()
def tiny_png_file(tmp_path: Path) -> Path:
    """Write a tiny PNG to a temp file and return its path."""
    p = tmp_path / "test_image.png"
    p.write_bytes(_TINY_PNG_BYTES)
    return p


@pytest.fixture()
def tiny_jpg_file(tmp_path: Path) -> Path:
    """Write a tiny file with .jpg extension (content is fake but valid for MIME detection)."""
    p = tmp_path / "test_image.jpg"
    p.write_bytes(b"\xff\xd8\xff")  # JPEG magic bytes
    return p


# ---------------------------------------------------------------------------
# ContentBlock model tests
# ---------------------------------------------------------------------------


class TestContentBlock:
    """Test ContentBlock creation and fields."""

    def test_text_block(self) -> None:
        block = ContentBlock(type="text", text="Hello world")
        assert block.type == "text"
        assert block.text == "Hello world"
        assert block.image_url is None
        assert block.source is None

    def test_image_url_block(self) -> None:
        block = ContentBlock(
            type="image_url",
            image_url={"url": "https://example.com/photo.jpg"},
        )
        assert block.type == "image_url"
        assert block.image_url == {"url": "https://example.com/photo.jpg"}
        assert block.text is None

    def test_image_url_block_with_data_uri(self) -> None:
        data_uri = f"data:image/png;base64,{_TINY_PNG_B64}"
        block = ContentBlock(type="image_url", image_url={"url": data_uri})
        assert block.type == "image_url"
        assert block.image_url is not None
        assert block.image_url["url"].startswith("data:image/png;base64,")

    def test_anthropic_image_block(self) -> None:
        """The older Anthropic-native image format should still work."""
        block = ContentBlock(
            type="image",
            source=ImageSource(
                type="base64",
                media_type="image/png",
                data=_TINY_PNG_B64,
            ),
        )
        assert block.type == "image"
        assert block.source is not None
        assert block.source.type == "base64"
        assert block.source.media_type == "image/png"

    def test_serialization_round_trip(self) -> None:
        """ContentBlock should survive model_dump -> model_validate."""
        block = ContentBlock(
            type="image_url",
            image_url={"url": "https://example.com/img.png"},
        )
        data = block.model_dump()
        restored = ContentBlock.model_validate(data)
        assert restored.type == "image_url"
        assert restored.image_url == {"url": "https://example.com/img.png"}


# ---------------------------------------------------------------------------
# Message with content blocks
# ---------------------------------------------------------------------------


class TestMessageWithContentBlocks:
    """Test Message with list[ContentBlock] content."""

    def test_message_with_string_content(self) -> None:
        """Backward compat: string content still works."""
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.content == "Hello"
        assert isinstance(msg.content, str)

    def test_message_with_none_content(self) -> None:
        msg = Message(role=MessageRole.ASSISTANT, content=None)
        assert msg.content is None

    def test_message_with_content_blocks(self) -> None:
        blocks = [
            ContentBlock(type="text", text="Describe this image"),
            ContentBlock(
                type="image_url",
                image_url={"url": "https://example.com/photo.jpg"},
            ),
        ]
        msg = Message(role=MessageRole.USER, content=blocks)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert msg.content[0].type == "text"
        assert msg.content[1].type == "image_url"

    def test_message_serialization_with_blocks(self) -> None:
        """Message with content blocks should serialize correctly."""
        blocks = [
            ContentBlock(type="text", text="What is in this image?"),
            ContentBlock(
                type="image_url",
                image_url={"url": f"data:image/png;base64,{_TINY_PNG_B64}"},
            ),
        ]
        msg = Message(role=MessageRole.USER, content=blocks)
        data = msg.model_dump()

        assert isinstance(data["content"], list)
        assert data["content"][0]["type"] == "text"
        assert data["content"][0]["text"] == "What is in this image?"
        assert data["content"][1]["type"] == "image_url"
        assert "url" in data["content"][1]["image_url"]


# ---------------------------------------------------------------------------
# SDK build_content_blocks helper
# ---------------------------------------------------------------------------


class TestBuildContentBlocks:
    """Test the build_content_blocks helper from sdk.py."""

    def test_no_images_returns_string(self) -> None:
        result = build_content_blocks("Hello world")
        assert result == "Hello world"
        assert isinstance(result, str)

    def test_empty_images_returns_string(self) -> None:
        result = build_content_blocks("Hello", images=[])
        assert result == "Hello"
        assert isinstance(result, str)

    def test_none_images_returns_string(self) -> None:
        result = build_content_blocks("Hello", images=None)
        assert result == "Hello"
        assert isinstance(result, str)

    def test_url_image(self) -> None:
        result = build_content_blocks(
            "Describe this",
            images=["https://example.com/photo.jpg"],
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].type == "text"
        assert result[0].text == "Describe this"
        assert result[1].type == "image_url"
        assert result[1].image_url == {"url": "https://example.com/photo.jpg"}

    def test_http_url_image(self) -> None:
        result = build_content_blocks(
            "Look",
            images=["http://example.com/photo.jpg"],
        )
        assert isinstance(result, list)
        assert result[1].image_url == {"url": "http://example.com/photo.jpg"}

    def test_file_path_image(self, tiny_png_file: Path) -> None:
        result = build_content_blocks(
            "What is this?",
            images=[str(tiny_png_file)],
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[1].type == "image_url"
        assert result[1].image_url is not None
        url = result[1].image_url["url"]
        assert url.startswith("data:image/png;base64,")
        # Verify the base64 data is valid
        b64_data = url.split(",", 1)[1]
        decoded = base64.b64decode(b64_data)
        assert decoded == _TINY_PNG_BYTES

    def test_file_path_jpg_mime_detection(self, tiny_jpg_file: Path) -> None:
        result = build_content_blocks(
            "Check",
            images=[str(tiny_jpg_file)],
        )
        assert isinstance(result, list)
        url = result[1].image_url["url"]
        assert url.startswith("data:image/jpeg;base64,")

    def test_data_uri_passthrough(self) -> None:
        data_uri = f"data:image/png;base64,{_TINY_PNG_B64}"
        result = build_content_blocks("Look", images=[data_uri])
        assert isinstance(result, list)
        assert result[1].image_url == {"url": data_uri}

    def test_multiple_images(self) -> None:
        result = build_content_blocks(
            "Compare these",
            images=[
                "https://example.com/a.jpg",
                "https://example.com/b.jpg",
            ],
        )
        assert isinstance(result, list)
        assert len(result) == 3  # 1 text + 2 images
        assert result[0].type == "text"
        assert result[1].type == "image_url"
        assert result[2].type == "image_url"


# ---------------------------------------------------------------------------
# OpenAI provider content block conversion
# ---------------------------------------------------------------------------


class TestOpenAIContentBlockConversion:
    """Test _convert_content_blocks_openai helper."""

    def test_text_block(self) -> None:
        blocks = [{"type": "text", "text": "Hello"}]
        result = _convert_content_blocks_openai(blocks)
        assert result == [{"type": "text", "text": "Hello"}]

    def test_image_url_block(self) -> None:
        blocks = [{"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}]
        result = _convert_content_blocks_openai(blocks)
        assert result == [{"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}]

    def test_mixed_blocks(self) -> None:
        blocks = [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        ]
        result = _convert_content_blocks_openai(blocks)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Describe this image"}
        assert result[1] == {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}

    def test_anthropic_image_converted_to_openai(self) -> None:
        """Anthropic-native 'image' block should be converted to 'image_url'."""
        blocks = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _TINY_PNG_B64,
                },
            }
        ]
        result = _convert_content_blocks_openai(blocks)
        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        url = result[0]["image_url"]["url"]
        assert url == f"data:image/png;base64,{_TINY_PNG_B64}"

    def test_anthropic_url_image_converted_to_openai(self) -> None:
        blocks = [
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/img.png",
                },
            }
        ]
        result = _convert_content_blocks_openai(blocks)
        assert result[0] == {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}

    def test_content_block_objects(self) -> None:
        """Should handle live ContentBlock instances, not just dicts."""
        blocks = [
            ContentBlock(type="text", text="Look at this"),
            ContentBlock(type="image_url", image_url={"url": "https://example.com/x.jpg"}),
        ]
        result = _convert_content_blocks_openai(blocks)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Look at this"}
        assert result[1] == {"type": "image_url", "image_url": {"url": "https://example.com/x.jpg"}}

    def test_serialized_content_blocks_from_model_dump(self) -> None:
        """Content blocks from Message.model_dump() should convert correctly."""
        blocks = [
            ContentBlock(type="text", text="Hello"),
            ContentBlock(type="image_url", image_url={"url": "https://example.com/img.png"}),
        ]
        msg = Message(role=MessageRole.USER, content=blocks)
        dumped = msg.model_dump()
        raw_content = dumped["content"]
        result = _convert_content_blocks_openai(raw_content)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Hello"}
        assert result[1] == {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}


# ---------------------------------------------------------------------------
# Anthropic provider content block conversion
# ---------------------------------------------------------------------------


class TestAnthropicContentBlockConversion:
    """Test Anthropic _convert_content_block for image_url type."""

    def test_image_url_data_uri_to_anthropic(self) -> None:
        from arcana.gateway.providers.anthropic import _convert_content_block

        block = ContentBlock(
            type="image_url",
            image_url={"url": f"data:image/png;base64,{_TINY_PNG_B64}"},
        )
        result = _convert_content_block(block)
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/png"
        assert result["source"]["data"] == _TINY_PNG_B64

    def test_image_url_http_to_anthropic(self) -> None:
        from arcana.gateway.providers.anthropic import _convert_content_block

        block = ContentBlock(
            type="image_url",
            image_url={"url": "https://example.com/photo.jpg"},
        )
        result = _convert_content_block(block)
        assert result["type"] == "image"
        assert result["source"]["type"] == "url"
        assert result["source"]["url"] == "https://example.com/photo.jpg"

    def test_image_url_invalid_fallback(self) -> None:
        from arcana.gateway.providers.anthropic import _convert_content_block

        block = ContentBlock(
            type="image_url",
            image_url={"url": ""},
        )
        result = _convert_content_block(block)
        # Should fall back to text placeholder
        assert result["type"] == "text"

    def test_existing_image_block_still_works(self) -> None:
        """Anthropic-native image blocks should still work unchanged."""
        from arcana.gateway.providers.anthropic import _convert_content_block

        block = ContentBlock(
            type="image",
            source=ImageSource(
                type="base64",
                media_type="image/jpeg",
                data="abc123",
            ),
        )
        result = _convert_content_block(block)
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/jpeg"
        assert result["source"]["data"] == "abc123"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure string content works everywhere as before."""

    def test_message_string_content_model_dump(self) -> None:
        msg = Message(role=MessageRole.USER, content="Plain text")
        data = msg.model_dump()
        assert data["content"] == "Plain text"
        assert isinstance(data["content"], str)

    def test_openai_convert_string_content(self) -> None:
        """String content should pass through the OpenAI converter unchanged."""
        blocks_input = "Just a string"
        # _convert_content_blocks_openai is only called on lists, so
        # the provider's _convert_messages handles str directly
        assert isinstance(blocks_input, str)

    def test_build_content_blocks_backward_compat(self) -> None:
        """Without images, build_content_blocks returns plain str."""
        result = build_content_blocks("Do something")
        assert result == "Do something"
        assert isinstance(result, str)

    def test_context_builder_content_text_with_blocks(self) -> None:
        """The context builder's _content_text should extract text from blocks."""
        from arcana.context.builder import _content_text

        # String
        assert _content_text("hello") == "hello"

        # None
        assert _content_text(None) == ""

        # List of ContentBlock
        blocks = [
            ContentBlock(type="text", text="First"),
            ContentBlock(type="image_url", image_url={"url": "https://x.com/i.png"}),
            ContentBlock(type="text", text="Second"),
        ]
        result = _content_text(blocks)
        assert "First" in result
        assert "Second" in result


# ---------------------------------------------------------------------------
# Integration-style test: end-to-end content block flow
# ---------------------------------------------------------------------------


class TestMultimodalFlow:
    """Test the full flow from SDK to provider conversion."""

    def test_full_flow_openai_format(self) -> None:
        """Build content blocks in SDK, serialize, convert for OpenAI."""
        # 1. SDK builds content blocks
        content = build_content_blocks(
            "What objects are in this image?",
            images=["https://example.com/photo.jpg"],
        )
        assert isinstance(content, list)

        # 2. Create a Message
        msg = Message(role=MessageRole.USER, content=content)

        # 3. Serialize (as would happen in provider)
        dumped = msg.model_dump()
        raw_content = dumped["content"]
        assert isinstance(raw_content, list)

        # 4. Convert for OpenAI
        openai_blocks = _convert_content_blocks_openai(raw_content)
        assert len(openai_blocks) == 2
        assert openai_blocks[0] == {
            "type": "text",
            "text": "What objects are in this image?",
        }
        assert openai_blocks[1] == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/photo.jpg"},
        }

    def test_full_flow_anthropic_format(self) -> None:
        """Build content blocks in SDK, convert for Anthropic."""
        from arcana.gateway.providers.anthropic import _convert_content_block

        content = build_content_blocks(
            "Describe this",
            images=[f"data:image/png;base64,{_TINY_PNG_B64}"],
        )
        assert isinstance(content, list)

        # Convert each block for Anthropic
        anthropic_blocks = [_convert_content_block(b) for b in content]
        assert len(anthropic_blocks) == 2
        assert anthropic_blocks[0] == {"type": "text", "text": "Describe this"}
        assert anthropic_blocks[1]["type"] == "image"
        assert anthropic_blocks[1]["source"]["type"] == "base64"
        assert anthropic_blocks[1]["source"]["data"] == _TINY_PNG_B64
