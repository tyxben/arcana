"""Eval — evaluation harness for agent testing and regression gating."""

from arcana.contracts.eval import (
    EvalCase,
    EvalReport,
    EvalResult,
    GateConfig,
    OutcomeCriterion,
    RegressionResult,
)
from arcana.eval.gate import RegressionGate
from arcana.eval.llm_judge import CriterionScore, JudgeResult, LLMJudge
from arcana.eval.runner import EvalRunner

__all__ = [
    # Contracts
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "GateConfig",
    "OutcomeCriterion",
    "RegressionResult",
    # Core
    "EvalRunner",
    "RegressionGate",
    # LLM Judge
    "CriterionScore",
    "JudgeResult",
    "LLMJudge",
]
