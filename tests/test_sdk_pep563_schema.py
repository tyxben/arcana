"""Regression tests for PEP-563 (``from __future__ import annotations``) tool schemas.

Under ``from __future__ import annotations`` -- the default in most modern user
modules -- function annotations are stored as *strings* (e.g. ``"int"`` rather
than the ``int`` type). Previously ``_signature_to_json_schema`` mapped these
string annotations through a ``type -> json-type`` dict, so they all fell
through to ``"string"`` and every typed tool parameter was schema'd as a string.
That then broke tool-argument validation at runtime (an ``int`` arg was rejected
as not-a-string).

This module MUST keep ``from __future__ import annotations`` at the top so the
annotations on the tools defined below are PEP-563 strings, exercising the fix.
"""

from __future__ import annotations

import arcana
from arcana.contracts.tool import ToolCall
from arcana.sdk import Tool, _FunctionToolProvider
from arcana.tool_gateway.gateway import ToolGateway
from arcana.tool_gateway.registry import ToolRegistry


class TestPep563ToolDecoratorSchema:
    def test_decorator_resolves_string_annotations_to_json_types(self):
        @arcana.tool()
        def typed_tool(
            a: int,
            b: float,
            c: bool,
            d: list,
            e: dict,
            f: str,
        ) -> str:
            return ""

        props = typed_tool._arcana_tool_spec.input_schema["properties"]
        assert props["a"]["type"] == "integer"
        assert props["b"]["type"] == "number"
        assert props["c"]["type"] == "boolean"
        assert props["d"]["type"] == "array"
        assert props["e"]["type"] == "object"
        assert props["f"]["type"] == "string"
        # Guard against the regression: nothing should silently be "string".
        assert props["a"]["type"] != "string"
        assert props["d"]["type"] != "string"

    def test_optional_int_resolves_to_integer(self):
        @arcana.tool()
        def opt_tool(x: int | None = None) -> str:
            return ""

        props = opt_tool._arcana_tool_spec.input_schema["properties"]
        assert props["x"]["type"] == "integer"

    def test_tool_class_resolves_string_annotations(self):
        def typed_fn(count: int, ratio: float) -> str:
            return ""

        t = Tool(fn=typed_fn)
        props = t._spec.input_schema["properties"]
        assert props["count"]["type"] == "integer"
        assert props["ratio"]["type"] == "number"


class TestPep563ToolGatewayValidation:
    async def _call(self, func, arguments) -> object:
        spec = func._arcana_tool_spec
        registry = ToolRegistry()
        registry.register(_FunctionToolProvider(spec, func))
        gateway = ToolGateway(registry=registry, granted_capabilities=set())
        call = ToolCall(id="c1", name=spec.name, arguments=arguments)
        return await gateway.call(call)

    async def test_int_arg_validates_end_to_end(self):
        @arcana.tool()
        def add_one(n: int) -> int:
            return n + 1

        result = await self._call(add_one, {"n": 41})
        assert result.success, result.error
        assert result.output == 42

    async def test_optional_int_arg_validates_end_to_end(self):
        @arcana.tool()
        def maybe(n: int | None = None) -> str:
            return f"got {n}"

        result = await self._call(maybe, {"n": 7})
        assert result.success, result.error
        assert result.output == "got 7"

    async def test_string_arg_still_rejected_for_int_param(self):
        # The schema is now correctly "integer", so a string arg must fail
        # validation -- proving the type is no longer "string".
        @arcana.tool()
        def needs_int(n: int) -> int:
            return n

        result = await self._call(needs_int, {"n": "not-an-int"})
        assert not result.success
        assert result.error is not None
        assert "integer" in result.error.message
