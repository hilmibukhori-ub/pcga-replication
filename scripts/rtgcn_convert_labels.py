"""Safely convert the RT-GCN released label pickles ({train,test}_15_label.pkl) to .npy.

The pickles are untrusted downloads. Two layers of protection:
1. A static opcode scan (pickletools). Every GLOBAL / STACK_GLOBAL target must be on a small allow-list
   of numpy array-reconstruction callables.
2. A restricted Unpickler whose find_class only resolves that same allow-list.
The result (a list of per-day arrays) is stacked to a float64 array and saved with allow_pickle=False.

usage: python -I scripts/rtgcn_convert_labels.py DATA_DIR   (e.g. data/raw/rtgcn_dl/NYSE)
"""
import pickle
import pickletools
import sys
from pathlib import Path

import numpy as np

ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("builtins", "list"),
}


def scan(raw):
    """Return the set of (module, name) globals referenced by the pickle; raise on anything else."""
    found, strings = set(), []
    for op, arg, _ in pickletools.genops(raw):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
            strings.append(arg)
        elif op.name == "GLOBAL":
            mod, name = arg.split(" ", 1)
            found.add((mod, name))
        elif op.name == "STACK_GLOBAL":
            found.add((strings[-2], strings[-1]))
        elif op.name in ("REDUCE", "BUILD", "NEWOBJ", "NEWOBJ_EX", "INST", "OBJ"):
            pass  # only reachable through an allowed global (checked below / by find_class)
    bad = found - ALLOWED
    if bad:
        raise RuntimeError(f"disallowed globals in pickle: {sorted(bad)}")
    return found


class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def main(d):
    d = Path(d)
    for part in ("train", "test"):
        src = d / f"{part}_15_label.pkl"
        raw = src.read_bytes()
        g = scan(raw)
        import io
        obj = Restricted(io.BytesIO(raw)).load()
        arr = np.stack([np.asarray(x, dtype=np.float64) for x in obj]) if isinstance(obj, (list, tuple)) else np.asarray(obj, np.float64)
        price = np.load(d / f"{part}_15_price.npy", allow_pickle=False)
        assert arr.shape == price.shape, (arr.shape, price.shape)
        assert np.isfinite(arr).all()
        np.save(d / f"{part}_15_label.npy", arr, allow_pickle=False)
        print(f"{src.name}: globals {sorted(g)} -> {arr.shape}, mean {arr.mean():+.2e}, sd {arr.std():.2e}")


if __name__ == "__main__":
    main(sys.argv[1])
