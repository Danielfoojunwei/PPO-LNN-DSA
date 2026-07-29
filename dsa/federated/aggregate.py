"""Parameter averaging and *measured* communication accounting.

Defect D7, part two.  The previous version of this repository did not measure the
bytes it moved.  It added them up in closed form at four call sites, producing a
flat arm of exactly 21.5005 MB and a hierarchical arm of exactly 28.6674 MB --
a ratio of exactly 4/3, with a reported across-seed standard deviation of exactly
0.0.  A quantity that is byte-identical across five seeds and lands on an exact
rational multiple of another is not a measurement; it is arithmetic wearing a
measurement's clothes.  (Byte counts *are* legitimately seed-invariant here -- the
tensors have the same shapes every time -- so the tell is not the zero variance on
its own.  The tell is that no tensor was ever consulted.)

The repair is structural rather than numerical: there is exactly one place in this
package where a byte count can come into existence, :func:`state_dict_bytes`, and
it reads ``numel() * element_size()`` off the actual tensors being transmitted.
:class:`CommunicationLedger` records a :class:`Transfer` at the moment a payload
crosses a link, and every reported aggregate is a sum over recorded transfers.
No script, table or document in this repository may compute a byte count any other
way; ``tests/test_federated.py::test_no_closed_form_bytes`` enforces that by
grepping this package for arithmetic on parameter sizes outside these two
functions.

Link taxonomy
-------------
======================  ==================================================
``cloud_up``            edge -> cloud (hierarchical), client -> cloud (flat)
``cloud_down``          cloud -> edge (hierarchical), cloud -> client (flat)
``edge_up``             client -> edge   (hierarchical only)
``edge_down``           edge   -> client (hierarchical only)
======================  ==================================================

``cloud_*`` transfers traverse the wide-area link and are the scarce resource the
hierarchy exists to conserve; ``edge_*`` transfers stay inside a local network
segment.  Both aggregates and their sum are reported in every federated table, so
the hierarchy's cost -- it moves *more* bytes in total -- is as visible as its
benefit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

__all__ = [
    "fedavg",
    "state_dict_bytes",
    "Transfer",
    "CommunicationLedger",
    "CLOUD_LINKS",
    "EDGE_LINKS",
    "LINK_KINDS",
    "BYTES_PER_MEGABYTE",
]

#: Links whose payloads traverse the wide-area (client/edge <-> cloud) path.
CLOUD_LINKS: tuple[str, ...] = ("cloud_up", "cloud_down")

#: Links whose payloads stay inside one edge's local network segment.
EDGE_LINKS: tuple[str, ...] = ("edge_up", "edge_down")

#: Every link kind the ledger will accept.
LINK_KINDS: tuple[str, ...] = CLOUD_LINKS + EDGE_LINKS

#: SI megabyte (10^6 bytes), not the binary MiB.  Stated because a factor of
#: 1.048576 silently applied to a headline number is exactly the sort of thing
#: this rebuild exists to make impossible to do by accident.
BYTES_PER_MEGABYTE: int = 1_000_000


# --------------------------------------------------------------------------- #
# Averaging
# --------------------------------------------------------------------------- #


def state_dict_bytes(sd: Mapping[str, torch.Tensor]) -> int:
    """Wire size of ``sd``, measured off the tensors themselves.

    This is the **only** function in the package permitted to turn tensors into a
    byte count.  It is deliberately trivial so that there is nowhere for a fudge
    factor to hide::

        sum(t.numel() * t.element_size() for t in sd.values())

    No compression, no sparsification, no quantisation is modelled: every arm
    transmits dense float32 parameters, so a byte-count comparison between arms is
    a comparison of *how many times* the parameters move and *over which link*,
    which is the only thing the topology actually changes.
    """
    total = 0
    for name, tensor in sd.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"payload entry {name!r} is {type(tensor).__name__}, not a torch.Tensor"
            )
        total += tensor.numel() * tensor.element_size()
    return int(total)


def fedavg(
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Weighted parameter average.

    Parameters
    ----------
    state_dicts
        Non-empty sequence of state dicts sharing an identical key set and, per
        key, identical shapes and dtypes.
    weights
        Non-negative mixing weights, normalised internally.  ``None`` means
        uniform.

    Notes
    -----
    Floating-point tensors are accumulated in ``float64`` and cast back to their
    original dtype, so the result does not depend on the order in which clients
    happen to be listed beyond float64 rounding.  Non-floating tensors (integer
    or boolean buffers) cannot be meaningfully averaged; they are required to be
    identical across contributors and are passed through unchanged.  If they ever
    disagree, that is a real modelling problem and this function raises rather
    than silently picking one.

    Optimizer state is **not** averaged and **not** carried across rounds: each
    local phase constructs a fresh optimizer from the received parameters.  That
    is standard FedAvg, and it is stated here because carrying Adam moments across
    a parameter average is a common silent deviation from the algorithm being
    claimed.
    """
    dicts = list(state_dicts)
    if not dicts:
        raise ValueError("fedavg requires at least one state dict")

    keys = list(dicts[0].keys())
    key_set = set(keys)
    for i, sd in enumerate(dicts[1:], start=1):
        if set(sd.keys()) != key_set:
            missing = sorted(key_set - set(sd.keys()))
            extra = sorted(set(sd.keys()) - key_set)
            raise ValueError(
                f"state dict {i} has a different key set (missing={missing}, extra={extra})"
            )

    if weights is None:
        w = [1.0] * len(dicts)
    else:
        w = [float(x) for x in weights]
        if len(w) != len(dicts):
            raise ValueError(
                f"got {len(w)} weights for {len(dicts)} state dicts"
            )
        if any(x < 0.0 for x in w):
            raise ValueError("weights must be non-negative")
    total_w = sum(w)
    if total_w <= 0.0:
        raise ValueError("weights must sum to a positive value")
    w = [x / total_w for x in w]

    out: dict[str, torch.Tensor] = {}
    for key in keys:
        ref = dicts[0][key]
        if ref.is_floating_point():
            acc = torch.zeros_like(ref, dtype=torch.float64)
            for weight, sd in zip(w, dicts):
                tensor = sd[key]
                if tensor.shape != ref.shape:
                    raise ValueError(
                        f"shape mismatch for {key!r}: {tuple(tensor.shape)} vs {tuple(ref.shape)}"
                    )
                acc += weight * tensor.detach().to(torch.float64)
            out[key] = acc.to(ref.dtype)
        else:
            for sd in dicts[1:]:
                if not torch.equal(sd[key], ref):
                    raise ValueError(
                        f"non-floating tensor {key!r} differs between clients and "
                        "cannot be averaged"
                    )
            out[key] = ref.detach().clone()
    return out


# --------------------------------------------------------------------------- #
# Communication ledger
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Transfer:
    """One payload crossing one link at one round.  Immutable once recorded."""

    link: str
    src: str
    dst: str
    num_bytes: int
    round_index: int

    def is_cloud(self) -> bool:
        return self.link in CLOUD_LINKS

    def to_row(self) -> dict[str, object]:
        return {
            "link": self.link,
            "src": self.src,
            "dst": self.dst,
            "num_bytes": int(self.num_bytes),
            "round_index": int(self.round_index),
        }


class CommunicationLedger:
    """An append-only log of every parameter payload a run put on a link.

    The ledger is the single source of truth for every communication number this
    repository reports.  A runner that forgets to call :meth:`record` under-reports
    its own cost; a runner that computes a cost without calling :meth:`record`
    cannot get that number into a table at all, because the table builder reads
    :meth:`totals`.
    """

    def __init__(self) -> None:
        self.transfers: list[Transfer] = []

    def __len__(self) -> int:
        return len(self.transfers)

    def record(
        self,
        link: str,
        src: str,
        dst: str,
        payload: Mapping[str, torch.Tensor],
        round_index: int,
    ) -> None:
        """Log one transfer, sizing it from the payload tensors."""
        if link not in LINK_KINDS:
            raise ValueError(f"link must be one of {LINK_KINDS}, got {link!r}")
        self.transfers.append(
            Transfer(
                link=link,
                src=str(src),
                dst=str(dst),
                num_bytes=state_dict_bytes(payload),
                round_index=int(round_index),
            )
        )

    # ------------------------------------------------------------ aggregates #

    def totals(self) -> dict[str, float]:
        """Aggregate byte counts.  Every federated table reports all three."""
        cloud = sum(t.num_bytes for t in self.transfers if t.is_cloud())
        edge = sum(t.num_bytes for t in self.transfers if not t.is_cloud())
        total = cloud + edge
        return {
            "cloud_bytes": float(cloud),
            "edge_local_bytes": float(edge),
            "total_bytes": float(total),
            "cloud_megabytes": cloud / BYTES_PER_MEGABYTE,
            "edge_local_megabytes": edge / BYTES_PER_MEGABYTE,
            "total_megabytes": total / BYTES_PER_MEGABYTE,
            "num_transfers": float(len(self.transfers)),
        }

    def by_link(self) -> dict[str, int]:
        """Bytes per link kind.  Absent link kinds report zero, not a missing key."""
        out = {kind: 0 for kind in LINK_KINDS}
        for t in self.transfers:
            out[t.link] += int(t.num_bytes)
        return out

    def counts_by_link(self) -> dict[str, int]:
        """Number of transfers per link kind."""
        out = {kind: 0 for kind in LINK_KINDS}
        for t in self.transfers:
            out[t.link] += 1
        return out

    def per_round(self) -> list[dict[str, float]]:
        """Cloud / edge / total bytes for each round index, in round order."""
        rounds = sorted({t.round_index for t in self.transfers})
        rows: list[dict[str, float]] = []
        for r in rounds:
            sel = [t for t in self.transfers if t.round_index == r]
            cloud = sum(t.num_bytes for t in sel if t.is_cloud())
            edge = sum(t.num_bytes for t in sel if not t.is_cloud())
            rows.append(
                {
                    "round_index": float(r),
                    "cloud_bytes": float(cloud),
                    "edge_local_bytes": float(edge),
                    "total_bytes": float(cloud + edge),
                    "num_transfers": float(len(sel)),
                }
            )
        return rows
