"""Chronological splits with label-maturity enforcement.

Timing convention (0-based internally):
  * Everything indexed ``t`` (features, factors, observed graph) is observed at the end of step t.
  * A prediction made at decision time t targets ``returns[t + h]`` (h = horizon), which is
    realised -- i.e. the label *matures* -- at time t + h.
  * A decision time t belongs to a split only if its label also matures inside that split
    (the last ``h`` decision times of each split are purged).
"""

from __future__ import annotations

import numpy as np

SPLIT_ORDER = ("expert_train", "expert_val", "meta_train", "selector_val", "calibration", "test")


class LeakageError(AssertionError):
    """Raised when an operation would use information that is not yet available."""


class TemporalSplits:
    def __init__(self, split_cfg: dict, n_steps: int):
        self.horizon = int(split_cfg["horizon"])
        self.n_steps = n_steps
        self._b: dict[str, tuple[int, int]] = {}
        prev_end = -1
        for name in SPLIT_ORDER:
            s1, e1 = split_cfg[name]
            s0, e0 = int(s1) - 1, int(e1) - 1
            if s0 != prev_end + 1:
                raise ValueError(f"split '{name}' is not contiguous with the previous split")
            if e0 - s0 + 1 <= self.horizon:
                raise ValueError(f"split '{name}' too short for horizon")
            self._b[name] = (s0, e0)
            prev_end = e0
        if prev_end != n_steps - 1:
            raise ValueError("splits must cover all time steps exactly")

    def start(self, name: str) -> int:
        return self._b[name][0]

    def end(self, name: str) -> int:
        return self._b[name][1]

    def decision_times(self, name: str) -> np.ndarray:
        """Decision times whose labels mature within the split (purged)."""
        s0, e0 = self._b[name]
        return np.arange(s0, e0 - self.horizon + 1)

    def split_of(self, t: int) -> str:
        for name, (s0, e0) in self._b.items():
            if s0 <= t <= e0:
                return name
        raise IndexError(t)

    # ---------------------------------------------------------------- guards
    def assert_labels_within(self, idx: np.ndarray, name: str) -> None:
        s0, e0 = self._b[name]
        idx = np.asarray(idx)
        if idx.size == 0:
            raise LeakageError(f"empty index for split '{name}'")
        if idx.min() < s0 or idx.max() + self.horizon > e0:
            raise LeakageError(
                f"decision times [{idx.min()}, {idx.max()}] have labels outside split '{name}' "
                f"[{s0}, {e0}] (horizon {self.horizon})"
            )

    def assert_labels_before(self, idx: np.ndarray, name: str) -> None:
        """All labels for these decision times mature strictly before split `name` starts."""
        idx = np.asarray(idx)
        if idx.size and idx.max() + self.horizon >= self.start(name):
            raise LeakageError(
                f"label at time {idx.max() + self.horizon} is not available before '{name}' "
                f"starts at {self.start(name)}"
            )

    def describe(self) -> dict:
        return {
            n: {"start_1based": s + 1, "end_1based": e + 1,
                "n_decision_times": len(self.decision_times(n))}
            for n, (s, e) in self._b.items()
        }
