"""Model registry and the capacity-matching solver.

Defect D11: in the old repository the two "liquid" winners were simply the two
biggest networks (parameter count vs reward correlated at Spearman 0.829), and the
ablation also differed by a whole transformer layer.  The fix has two parts:

1. **Equal depth.**  Every registered model is exactly two blocks, so
   ``ppo_ltc_cfc`` differs from ``ppo_ltc`` in exactly one block and from
   ``ppo_cfc`` in exactly one block.
2. **Equal capacity.**  The hidden width ``H`` is *solved* per model so that every
   model lands within ``PARAM_TOLERANCE`` of ``PARAM_TARGET``.  No width is ever
   hardcoded, and the solved widths are written to ``results/manifest.json`` by the
   runner rather than into source.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .cells import ATTENTION_WINDOW
from .policy import RecurrentActorCritic

__all__ = [
    "PARAM_TARGET",
    "PARAM_TOLERANCE",
    "ModelSpec",
    "MODEL_REGISTRY",
    "PRIMARY_MODELS",
    "ATTENTION_HEAD_CANDIDATES",
    "solve_hidden_dim",
    "solve_width_and_heads",
    "build_model",
    "count_parameters",
    "model_summary",
]

PARAM_TARGET: int = 40_000
PARAM_TOLERANCE: float = 0.05

#: Head counts the joint solver considers for attention models, most-heads-first.
#: ``nn.MultiheadAttention`` requires ``embed_dim % num_heads == 0``, so the search
#: over ``H`` is restricted to multiples of the candidate head count.
ATTENTION_HEAD_CANDIDATES: tuple[int, ...] = (4, 2, 1)

_MAX_HIDDEN_DIM = 1024


@dataclass(frozen=True)
class ModelSpec:
    key: str
    cell_kinds: tuple[str, str]
    dt_aware: bool
    family: str
    cell_kwargs: dict = field(default_factory=dict)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "ppo_mlp": ModelSpec("ppo_mlp", ("mlp", "mlp"), dt_aware=False, family="memoryless"),
    "ppo_gru": ModelSpec("ppo_gru", ("gru", "gru"), dt_aware=False, family="gated_rnn"),
    "ppo_lstm": ModelSpec("ppo_lstm", ("lstm", "lstm"), dt_aware=False, family="gated_rnn"),
    "ppo_transformer": ModelSpec(
        "ppo_transformer", ("attn", "attn"), dt_aware=False, family="attention"
    ),
    "ppo_ltc": ModelSpec("ppo_ltc", ("ltc", "ltc"), dt_aware=True, family="liquid"),
    "ppo_cfc": ModelSpec("ppo_cfc", ("cfc", "cfc"), dt_aware=True, family="liquid"),
    "ppo_cfc_dtblind": ModelSpec(
        "ppo_cfc_dtblind",
        ("cfc_dtblind", "cfc_dtblind"),
        dt_aware=False,
        family="liquid_control",
    ),
    "ppo_ltc_cfc": ModelSpec("ppo_ltc_cfc", ("ltc", "cfc"), dt_aware=True, family="liquid"),
}

#: ``ppo_lstm`` is registered, capacity-matched and unit-tested, but is deliberately
#: not in the confirmatory matrix: it is a near-duplicate control of ``ppo_gru`` and
#: is dropped to buy statistical power.  It remains runnable via ``--models ppo_lstm``.
PRIMARY_MODELS: tuple[str, ...] = (
    "ppo_mlp",
    "ppo_gru",
    "ppo_transformer",
    "ppo_ltc",
    "ppo_cfc",
    "ppo_cfc_dtblind",
    "ppo_ltc_cfc",
)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _uses_attention(spec: ModelSpec) -> bool:
    return any(kind == "attn" for kind in spec.cell_kinds)


def _construct(
    spec: ModelSpec,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int,
    num_heads: int | None,
    generator: torch.Generator | None,
) -> RecurrentActorCritic:
    """Build one model, never disturbing the global RNG.

    ``nn.Linear.__init__`` calls ``reset_parameters`` which draws from the *global*
    torch RNG before :func:`deterministic_init_` overwrites everything.  Forking the
    RNG here keeps model construction free of global side effects, which principle P4
    (determinism independent of call order) requires.
    """
    kwargs = dict(spec.cell_kwargs)
    if num_heads is not None:
        kwargs.setdefault("num_heads", num_heads)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return RecurrentActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            cell_kinds=spec.cell_kinds,
            hidden_dim=hidden_dim,
            generator=generator,
            cell_kwargs=kwargs,
        )


def _param_count(
    spec: ModelSpec, obs_dim: int, action_dim: int, hidden_dim: int, num_heads: int | None
) -> int:
    return count_parameters(_construct(spec, obs_dim, action_dim, hidden_dim, num_heads, None))


def _solve_width_for_heads(
    spec: ModelSpec, obs_dim: int, action_dim: int, target: int, num_heads: int | None
) -> int:
    """Smallest-error ``H`` restricted to multiples of ``num_heads`` (or any ``H``).

    ``P(H)`` is monotonically increasing for every registered cell family (it is a
    positive quadratic in ``H`` with positive linear term), so we bracket the crossing
    by doubling, bisect for it, then check the three integers around it and return the
    argmin of ``|P(H) - target|``.  Ties resolve to the smaller ``H``.
    """
    step = 1 if num_heads is None else int(num_heads)

    def params_at(units: int) -> int:
        return _param_count(spec, obs_dim, action_dim, units * step, num_heads)

    high = 1
    while params_at(high) < target and (high * 2) * step <= _MAX_HIDDEN_DIM:
        high *= 2
    low = max(1, high // 2)
    while low < high:
        mid = (low + high) // 2
        if params_at(mid) >= target:
            high = mid
        else:
            low = mid + 1

    candidates = [c for c in (low - 1, low, low + 1) if c >= 1 and c * step <= _MAX_HIDDEN_DIM]
    best = min(candidates, key=lambda c: (abs(params_at(c) - target), c))
    return best * step


def solve_width_and_heads(
    spec: ModelSpec, obs_dim: int, action_dim: int, target: int = PARAM_TARGET
) -> tuple[int, int | None]:
    """Return ``(hidden_dim, num_heads)``.  ``num_heads`` is ``None`` for non-attention.

    For attention specs the search is joint over ``(H, num_heads)`` with
    ``num_heads in ATTENTION_HEAD_CANDIDATES`` and ``H % num_heads == 0``; among the
    feasible points it minimises ``|P - target|`` and breaks ties toward the larger
    head count.  Head count is not a factor under study -- it is solved, not chosen.
    """
    if not _uses_attention(spec):
        return _solve_width_for_heads(spec, obs_dim, action_dim, target, None), None

    best: tuple[int, int, int] | None = None  # (error, H, num_heads)
    for heads in ATTENTION_HEAD_CANDIDATES:  # descending -> larger head count wins ties
        hidden = _solve_width_for_heads(spec, obs_dim, action_dim, target, heads)
        error = abs(_param_count(spec, obs_dim, action_dim, hidden, heads) - target)
        if best is None or error < best[0]:
            best = (error, hidden, heads)
    assert best is not None
    return best[1], best[2]


def solve_hidden_dim(
    spec: ModelSpec, obs_dim: int, action_dim: int, target: int = PARAM_TARGET
) -> int:
    """Smallest-error integer hidden width for ``spec``.  Pure function.

    No globals, no caching that depends on call order.  See
    :func:`solve_width_and_heads` for the joint attention search; the solved head
    count is exposed through :func:`model_summary`.
    """
    return solve_width_and_heads(spec, obs_dim, action_dim, target)[0]


def build_model(
    key: str, obs_dim: int, action_dim: int, seed: int
) -> RecurrentActorCritic:
    """Deterministic: same ``(key, obs_dim, action_dim, seed)`` -> identical ``state_dict``.

    Initialisation uses only an explicit ``torch.Generator``; ``torch.manual_seed`` is
    never called on the global RNG (spec section 9.2).
    """
    if key not in MODEL_REGISTRY:
        raise KeyError(f"unknown model key {key!r}; known keys: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[key]
    hidden_dim, num_heads = solve_width_and_heads(spec, obs_dim, action_dim)
    generator = torch.Generator()
    generator.manual_seed(int(seed) % (2**63))
    return _construct(spec, obs_dim, action_dim, hidden_dim, num_heads, generator)


def model_summary(key: str, obs_dim: int, action_dim: int) -> dict[str, object]:
    """Static description of one registry entry.  Nothing here is hardcoded."""
    spec = MODEL_REGISTRY[key]
    hidden_dim, num_heads = solve_width_and_heads(spec, obs_dim, action_dim)
    model = _construct(spec, obs_dim, action_dim, hidden_dim, num_heads, None)
    parameter_count = count_parameters(model)
    return {
        "key": key,
        "hidden_dim": int(hidden_dim),
        "num_heads": None if num_heads is None else int(num_heads),
        "attention_window": ATTENTION_WINDOW if _uses_attention(spec) else None,
        "parameter_count": parameter_count,
        "param_error_frac": (parameter_count - PARAM_TARGET) / PARAM_TARGET,
        "cell_kinds": list(spec.cell_kinds),
        "dt_aware": bool(spec.dt_aware),
        "family": spec.family,
        "state_sizes": list(model.state_sizes),
    }
