"""Multi-model router for intelligent model selection by role and complexity."""

from __future__ import annotations

from arcana.contracts.llm import LLMRequest, LLMResponse, ModelConfig
from arcana.contracts.routing import (
    ModelRole,
    RoutingConfig,
    TaskComplexity,
)
from arcana.contracts.trace import TraceContext
from arcana.gateway.registry import ModelGatewayRegistry


class ModelRouter:
    """
    根据任务角色和复杂度选择最优模型。

    包装 ModelGatewayRegistry，不替代它。
    """

    def __init__(
        self,
        registry: ModelGatewayRegistry,
        config: RoutingConfig | None = None,
    ) -> None:
        self._registry = registry
        self._config = config or RoutingConfig()

    def get_config_for_role(self, role: ModelRole) -> ModelConfig:
        """为指定角色返回 ModelConfig。

        If the model is None in the routing config, resolve it from the
        provider's default_model attribute. Raise ValueError if no default
        can be determined.
        """
        role_map: dict[ModelRole, tuple[str, str | None]] = {
            ModelRole.ROUTER: (
                self._config.router_provider,
                self._config.router_model,
            ),
            ModelRole.STRATEGIST: (
                self._config.strategist_provider,
                self._config.strategist_model,
            ),
            ModelRole.EXECUTOR: (
                self._config.executor_provider,
                self._config.executor_model,
            ),
            ModelRole.COMPRESSOR: (
                self._config.compressor_provider,
                self._config.compressor_model,
            ),
            ModelRole.VALIDATOR: (
                self._config.validator_provider,
                self._config.validator_model,
            ),
        }
        provider_name, model_id = role_map[role]
        if not model_id:
            provider = self._registry.get(provider_name)
            if provider and hasattr(provider, "default_model"):
                dm = provider.default_model
                if isinstance(dm, str) and dm:
                    model_id = dm
            if not model_id:
                msg = (
                    f"No model configured for role '{role.value}' "
                    f"and provider '{provider_name}' has no default_model. "
                    "Set the model explicitly in RoutingConfig."
                )
                raise ValueError(msg)
        return ModelConfig(provider=provider_name, model_id=model_id)

    def select_role(
        self,
        step_type: str,
        complexity: TaskComplexity | None = None,
    ) -> ModelRole:
        """根据步骤类型和复杂度选择模型角色。

        Args:
            step_type: 步骤类型，例如 "think", "plan", "act", "verify", "compress"。
            complexity: 任务复杂度（可选）。

        Returns:
            选定的 ModelRole。
        """
        # 策略推理（think/plan）→ STRATEGIST
        if step_type in {"think", "plan"}:
            # 简单任务自动降级到 EXECUTOR
            if (
                self._config.auto_downgrade
                and complexity is not None
                and complexity in {TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE}
            ):
                return ModelRole.EXECUTOR
            return ModelRole.STRATEGIST

        # 工具执行（act）→ EXECUTOR
        if step_type == "act":
            return ModelRole.EXECUTOR

        # 验证（verify）→ VALIDATOR
        if step_type == "verify":
            return ModelRole.VALIDATOR

        # 压缩（compress）→ COMPRESSOR
        if step_type == "compress":
            return ModelRole.COMPRESSOR

        # 路由/分类（route/classify）→ ROUTER
        if step_type in {"route", "classify"}:
            return ModelRole.ROUTER

        # 默认使用 EXECUTOR
        return ModelRole.EXECUTOR

    def estimate_complexity(
        self,
        goal: str,
        step_count: int = 0,
        error_count: int = 0,
    ) -> TaskComplexity:
        """估算任务复杂度。纯函数。

        Args:
            goal: 任务目标描述。
            step_count: 已执行步骤数。
            error_count: 累计错误数。

        Returns:
            估算的 TaskComplexity。
        """
        complex_keywords = [
            "设计",
            "架构",
            "分析",
            "对比",
            "design",
            "architect",
            "analyze",
            "compare",
            "debug",
        ]
        simple_keywords = [
            "什么是",
            "解释",
            "翻译",
            "what is",
            "explain",
            "translate",
            "convert",
        ]

        score = 0
        goal_lower = goal.lower()
        for kw in complex_keywords:
            if kw in goal_lower:
                score += 2
        for kw in simple_keywords:
            if kw in goal_lower:
                score -= 1

        # 长度因素
        if len(goal) > 500:
            score += 1

        # 步骤历史
        score += step_count // 3
        score += error_count

        if score <= 0:
            return TaskComplexity.TRIVIAL
        if score <= 2:
            return TaskComplexity.SIMPLE
        if score <= 4:
            return TaskComplexity.MODERATE
        if score <= 6:
            return TaskComplexity.COMPLEX
        return TaskComplexity.EXPERT

    async def generate(
        self,
        request: LLMRequest,
        role: ModelRole,
        trace_ctx: TraceContext | None = None,
    ) -> LLMResponse:
        """通过角色路由生成响应。

        Args:
            request: LLM 请求。
            role: 模型角色。
            trace_ctx: 可选的 Trace 上下文。

        Returns:
            LLM 响应。
        """
        config = self.get_config_for_role(role)
        return await self._registry.generate(request, config, trace_ctx)
