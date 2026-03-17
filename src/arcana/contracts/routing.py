"""Model routing contracts and data models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class ModelRole(str, Enum):
    """模型在 agent 执行中的角色。"""

    ROUTER = "router"  # 路由/分类（小模型）
    STRATEGIST = "strategist"  # 策略推理（大模型）
    EXECUTOR = "executor"  # 工具调用和执行（中模型）
    COMPRESSOR = "compressor"  # 摘要/压缩/提取（小模型）
    VALIDATOR = "validator"  # 质量检查/校验（中模型）


class TaskComplexity(str, Enum):
    """任务复杂度等级。"""

    TRIVIAL = "trivial"  # 直接回答
    SIMPLE = "simple"  # 1-2 步
    MODERATE = "moderate"  # 3-5 步
    COMPLEX = "complex"  # 6+ 步，需要规划
    EXPERT = "expert"  # 需要深度推理


class RoutingConfig(BaseModel):
    """模型路由配置。可在 YAML 中声明。"""

    model_config = {"protected_namespaces": ()}

    # 每个角色对应的 provider + model
    router_provider: str = "deepseek"
    router_model: str = "deepseek-chat"
    strategist_provider: str = "anthropic"
    strategist_model: str = "claude-sonnet-4-20250514"
    executor_provider: str = "deepseek"
    executor_model: str = "deepseek-chat"
    compressor_provider: str = "deepseek"
    compressor_model: str = "deepseek-chat"
    validator_provider: str = "deepseek"
    validator_model: str = "deepseek-chat"

    # 优化选项
    auto_downgrade: bool = True  # 简单任务自动用小模型
    complexity_threshold: float = 0.6  # 超过此阈值升级到大模型


class RoutingDecision(BaseModel):
    """一次路由决策的记录（用于 Trace）。"""

    role: ModelRole
    selected_provider: str
    selected_model: str
    task_complexity: TaskComplexity | None = None
    reason: str = ""
