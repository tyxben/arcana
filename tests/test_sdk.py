"""Tests for SDK public API: @tool, run(), _FunctionToolProvider."""

import inspect

import pytest

from arcana.contracts.tool import SideEffect, ToolCall
from arcana.sdk import (
    RunResult,
    _FunctionToolProvider,
    _signature_to_json_schema,
    tool,
)


class TestToolDecorator:
    def test_basic(self):
        @tool()
        def my_func(x: str) -> str:
            return x

        assert hasattr(my_func, "_arcana_tool_spec")
        spec = my_func._arcana_tool_spec
        assert spec.name == "my_func"

    def test_custom_name(self):
        @tool(name="custom")
        def func(x: str) -> str:
            return x

        assert func._arcana_tool_spec.name == "custom"

    def test_description_from_docstring(self):
        @tool()
        def func(x: str) -> str:
            """My docstring."""
            return x

        assert func._arcana_tool_spec.description == "My docstring."

    def test_description_fallback(self):
        @tool()
        def func(x: str) -> str:
            return x

        assert "func" in func._arcana_tool_spec.description

    def test_custom_description(self):
        @tool(description="Custom desc")
        def func(x: str) -> str:
            return x

        assert func._arcana_tool_spec.description == "Custom desc"

    def test_affordance_fields(self):
        @tool(when_to_use="for math", what_to_expect="a number")
        def calc(expr: str) -> str:
            return expr

        spec = calc._arcana_tool_spec
        assert spec.when_to_use == "for math"
        assert spec.what_to_expect == "a number"

    def test_failure_meaning(self):
        @tool(failure_meaning="expression was invalid")
        def calc(expr: str) -> str:
            return expr

        assert calc._arcana_tool_spec.failure_meaning == "expression was invalid"

    def test_side_effect_read(self):
        @tool()
        def reader(path: str) -> str:
            return path

        assert reader._arcana_tool_spec.side_effect == SideEffect.READ

    def test_side_effect_write(self):
        @tool(side_effect="write")
        def writer(path: str) -> str:
            return path

        assert writer._arcana_tool_spec.side_effect == SideEffect.WRITE

    def test_side_effect_none(self):
        @tool(side_effect="none")
        def pure(x: str) -> str:
            return x

        assert pure._arcana_tool_spec.side_effect == SideEffect.NONE

    def test_requires_confirmation(self):
        @tool(requires_confirmation=True)
        def dangerous(x: str) -> str:
            return x

        assert dangerous._arcana_tool_spec.requires_confirmation is True

    def test_schema_generation(self):
        @tool()
        def func(name: str, count: int) -> str:
            return ""

        schema = func._arcana_tool_spec.input_schema
        assert "name" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["count"]["type"] == "integer"
        assert "name" in schema["required"]
        assert "count" in schema["required"]

    def test_callable_preserved(self):
        @tool()
        def my_func(x: str) -> str:
            return f"result: {x}"

        assert my_func("hello") == "result: hello"


class TestSignatureToJsonSchema:
    def test_basic_types(self):
        def f(a: str, b: int, c: float, d: bool):
            pass

        schema = _signature_to_json_schema(inspect.signature(f))
        assert schema["properties"]["a"]["type"] == "string"
        assert schema["properties"]["b"]["type"] == "integer"
        assert schema["properties"]["c"]["type"] == "number"
        assert schema["properties"]["d"]["type"] == "boolean"

    def test_list_and_dict(self):
        def f(items: list, data: dict):
            pass

        schema = _signature_to_json_schema(inspect.signature(f))
        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["data"]["type"] == "object"

    def test_optional_params(self):
        def f(required: str, optional: str = "default"):
            pass

        schema = _signature_to_json_schema(inspect.signature(f))
        assert "required" in schema.get("required", [])
        assert "optional" not in schema.get("required", [])

    def test_no_required_when_all_optional(self):
        def f(a: str = "x", b: int = 0):
            pass

        schema = _signature_to_json_schema(inspect.signature(f))
        assert "required" not in schema

    def test_skips_self(self):
        def f(self, x: str):
            pass

        schema = _signature_to_json_schema(inspect.signature(f))
        assert "self" not in schema["properties"]
        assert "x" in schema["properties"]

    def test_skips_cls(self):
        def f(cls, x: str):
            pass

        schema = _signature_to_json_schema(inspect.signature(f))
        assert "cls" not in schema["properties"]
        assert "x" in schema["properties"]

    def test_unannotated_defaults_to_string(self):
        def f(x):
            pass

        schema = _signature_to_json_schema(inspect.signature(f))
        assert schema["properties"]["x"]["type"] == "string"

    def test_empty_function(self):
        def f():
            pass

        schema = _signature_to_json_schema(inspect.signature(f))
        assert schema["properties"] == {}
        assert "required" not in schema


class TestFunctionToolProvider:
    @pytest.mark.asyncio
    async def test_sync_function(self):
        @tool()
        def add(a: str) -> str:
            return f"got {a}"

        provider = _FunctionToolProvider(spec=add._arcana_tool_spec, func=add)
        call = ToolCall(id="1", name="add", arguments={"a": "hello"})
        result = await provider.execute(call)
        assert result.success
        assert result.output == "got hello"

    @pytest.mark.asyncio
    async def test_async_function(self):
        @tool()
        async def async_add(a: str) -> str:
            return f"async {a}"

        provider = _FunctionToolProvider(
            spec=async_add._arcana_tool_spec, func=async_add
        )
        call = ToolCall(id="1", name="async_add", arguments={"a": "hi"})
        result = await provider.execute(call)
        assert result.success
        assert result.output == "async hi"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        @tool()
        def fail(x: str) -> str:
            raise ValueError("boom")

        provider = _FunctionToolProvider(spec=fail._arcana_tool_spec, func=fail)
        call = ToolCall(id="1", name="fail", arguments={"x": ""})
        result = await provider.execute(call)
        assert not result.success
        assert result.error is not None
        assert "boom" in result.error.message

    @pytest.mark.asyncio
    async def test_async_error_handling(self):
        @tool()
        async def fail_async(x: str) -> str:
            raise RuntimeError("async boom")

        provider = _FunctionToolProvider(
            spec=fail_async._arcana_tool_spec, func=fail_async
        )
        call = ToolCall(id="1", name="fail_async", arguments={"x": ""})
        result = await provider.execute(call)
        assert not result.success
        assert result.error is not None
        assert "async boom" in result.error.message

    @pytest.mark.asyncio
    async def test_health_check(self):
        @tool()
        def f(x: str) -> str:
            return x

        provider = _FunctionToolProvider(spec=f._arcana_tool_spec, func=f)
        assert await provider.health_check()

    @pytest.mark.asyncio
    async def test_tool_call_id_preserved(self):
        @tool()
        def echo(x: str) -> str:
            return x

        provider = _FunctionToolProvider(spec=echo._arcana_tool_spec, func=echo)
        call = ToolCall(id="call-42", name="echo", arguments={"x": "test"})
        result = await provider.execute(call)
        assert result.tool_call_id == "call-42"
        assert result.name == "echo"

    def test_spec_property(self):
        @tool(name="my_tool")
        def f(x: str) -> str:
            return x

        provider = _FunctionToolProvider(spec=f._arcana_tool_spec, func=f)
        assert provider.spec.name == "my_tool"


class TestRunResultSerialization:
    def test_serialization(self):
        r = RunResult(output="test", success=True, steps=1)
        data = r.model_dump()
        assert data["output"] == "test"
        assert data["success"] is True
        assert data["steps"] == 1

    def test_roundtrip(self):
        r = RunResult(
            output="hello",
            success=True,
            steps=5,
            tokens_used=200,
            cost_usd=0.01,
            run_id="abc-123",
        )
        data = r.model_dump()
        r2 = RunResult(**data)
        assert r2.output == r.output
        assert r2.success == r.success
        assert r2.steps == r.steps
        assert r2.tokens_used == r.tokens_used
        assert r2.cost_usd == r.cost_usd
        assert r2.run_id == r.run_id
