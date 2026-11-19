"""Timing / feasibility probe for RT-GCN (Zheng et al. 2023) on CPU. Imports the ORIGINAL model code from
data/raw/rtgcn_repo unchanged; only Tensor.cuda is patched to a no-op (CPU). Loss re-implemented 1:1
(net/loss.py hard-codes device='cuda'). Usage: python scripts/rtgcn_timing.py {Uniform|Time-aware} [batches]"""
import sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "data/raw/rtgcn_repo"
D = ROOT / "data/raw/rtgcn_dl/NASDAQ"
sys.path.insert(0, str(REPO))
torch.Tensor.cuda = lambda self, *a, **k: self
from net.rt_gcn import Model  # noqa: E402


def loss_fn(pred, gt, alpha):
    N, V = gt.size()
    p, g = pred.view(N, V, 1), gt.view(N, V, 1)
    one = torch.ones((V, 1))
    pp = p @ one.T - one @ p.permute(0, 2, 1)
    gg = one @ g.permute(0, 2, 1) - g @ one.T
    return torch.nn.functional.mse_loss(p, g) + alpha * torch.relu(pp * gg).mean()


mode = sys.argv[1]
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 5
BS = int(sys.argv[3]) if len(sys.argv) > 3 else 64
torch.set_num_threads(2)
X = np.load(D / "train_15_EOD.npy", mmap_mode="r")
P = np.load(D / "train_15_price.npy", mmap_mode="r")
Y = np.load(D / "train_15_label.npy")
m = Model(in_channels=4, graph_args={"market": "NASDAQ", "relation_path": str(D / "NASDAQ_relation.npy"), "strategy": "uniform"},
          edge_importance_weighting=mode, dropout=0.5)
opt = torch.optim.Adam(m.parameters(), lr=0.1, weight_decay=0.1)   # released train.yaml
print("params", sum(p.numel() for p in m.parameters()))
idx = np.random.default_rng(0).permutation(len(Y))
t0 = time.time()
for b in range(nb):
    j = np.sort(idx[b * BS:(b + 1) * BS])
    x = torch.from_numpy(np.ascontiguousarray(X[j][:, [0, 1, 2, 4]], dtype=np.float32))
    cp = torch.from_numpy(np.ascontiguousarray(P[j])); y = torch.from_numpy(Y[j])
    out = m(x)
    loss = loss_fn((out - cp) / cp, y, 0.1)
    opt.zero_grad(); loss.backward(); opt.step()
    if b == 0:
        t1 = time.time()
dt = (time.time() - t1) / max(nb - 1, 1)
print(f"{mode}: {dt:.2f}s per batch of {BS} -> ~{dt * len(Y) / BS / 60:.1f} min per epoch (1295 samples); loss {loss.item():.4g}")
import resource; print("max RSS GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)
