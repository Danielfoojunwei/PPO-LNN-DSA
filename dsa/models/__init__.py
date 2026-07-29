"""Capacity-matched, equal-depth recurrent actor-critic models.

Vocabulary (spec section 1.1).  Liquid Foundation Models are built on the
closed-form continuous-time (CfC) cell of Hasani et al. (2022); in this repository
the "LFM" component of the title is :class:`~dsa.models.cells.CfCCell`, registered as
``ppo_cfc``.  The attention block that a previous version of this repository called
"LFM" is now :class:`~dsa.models.cells.CausalAttentionBlock`, registered as
``ppo_transformer``, and it is a baseline -- not a liquid model.
"""

from .cells import (
    ATTENTION_WINDOW,
    CELL_REGISTRY,
    DEFAULT_LTC_UNFOLDS,
    DEFAULT_TAU_MAX,
    DEFAULT_TAU_MIN,
    CausalAttentionBlock,
    CfCCell,
    CfCDtBlindCell,
    GRUBlock,
    LSTMBlock,
    LTCCell,
    MLPBlock,
    RecurrentBlock,
    RecurrentBlockBase,
)
from .policy import RecurrentActorCritic, deterministic_init_
from .registry import (
    ATTENTION_HEAD_CANDIDATES,
    MODEL_REGISTRY,
    PARAM_TARGET,
    PARAM_TOLERANCE,
    PRIMARY_MODELS,
    ModelSpec,
    build_model,
    count_parameters,
    model_summary,
    solve_hidden_dim,
    solve_width_and_heads,
)

__all__ = [
    "ATTENTION_HEAD_CANDIDATES",
    "ATTENTION_WINDOW",
    "CELL_REGISTRY",
    "DEFAULT_LTC_UNFOLDS",
    "DEFAULT_TAU_MAX",
    "DEFAULT_TAU_MIN",
    "MODEL_REGISTRY",
    "PARAM_TARGET",
    "PARAM_TOLERANCE",
    "PRIMARY_MODELS",
    "CausalAttentionBlock",
    "CfCCell",
    "CfCDtBlindCell",
    "GRUBlock",
    "LSTMBlock",
    "LTCCell",
    "MLPBlock",
    "ModelSpec",
    "RecurrentActorCritic",
    "RecurrentBlock",
    "RecurrentBlockBase",
    "build_model",
    "count_parameters",
    "deterministic_init_",
    "model_summary",
    "solve_hidden_dim",
    "solve_width_and_heads",
]
