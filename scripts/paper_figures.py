"""Result figures for the ESWA paper (static PDF, light/print).

Fig. edge: real graph vs within-industry placebos at seed 0 (RSR NASDAQ, RSR NYSE, RT-GCN(U) NASDAQ).
Fig. D1:   per-seed paired differences real - no-relation, with mean, 95% t-interval and the published gain.

Reads only finished outputs (no training). Run with any Python that has numpy + matplotlib + scipy:
    /opt/anaconda3/bin/python scripts/paper_figures.py
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RSR = ROOT / "outputs/rsr_replica_v1"
RT = ROOT / "outputs/rtgcn_v1"
EXT = ROOT / "outputs/rtgcn_ext_v1"
FIG = ROOT / "paper/figures"
FIG.mkdir(parents=True, exist_ok=True)

# Palette: reference categorical slots 1-2 (validated pair) + neutral ink/grays
REAL = "#2a78d6"      # series 1: real relation graph
NONE = "#eb6834"      # series 2: no-relation model
PLAC = "#a9a8a3"      # neutral: placebos / per-seed dots
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})


def rsr_irr(market, graph, seed):
    return float(np.load(RSR / f"{market}_wikidata_{graph}_s{seed}.npz")["test_top1"].sum())


def rt_irr_val(graph, seed=0, cell="NASDAQ_U"):
    r = np.load(RT / f"NASDAQ_val_paper_{graph}_s{seed}.npz") if cell == "NASDAQ_U" else np.load(EXT / f"{cell}_val_paper_{graph}_s{seed}.npz")
    e = int(np.argmin(r["val_loss"]))
    return float(r["test_top1"][e].sum())


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Fig. edge
panels = []
for m in ("NASDAQ", "NYSE"):
    plac = np.array([rsr_irr(m, f"perm{k:03d}", 0) for k in range(99)])
    panels.append((f"RSR, {m} (99 placebos)", plac, rsr_irr(m, "real", 0), rsr_irr(m, "none", 0)))
plac = np.array([rt_irr_val(f"perm{k:03d}") for k in range(19)])
panels.append(("RT-GCN(U), NASDAQ (19 placebos)", plac, rt_irr_val("real"), rt_irr_val("none")))
plac = np.array([rt_irr_val(f"perm{k:03d}", cell="NASDAQ_T") for k in range(19)])
panels.append(("RT-GCN(T), NASDAQ (19 placebos)", plac, rt_irr_val("real", cell="NASDAQ_T"), rt_irr_val("none", cell="NASDAQ_T")))

fig, axes = plt.subplots(4, 1, figsize=(6.3, 5.4), gridspec_kw={"hspace": 1.05})
rng = np.random.default_rng(1)
for ax, (title, plac, real, none) in zip(axes, panels):
    rank = 1 + int((plac >= real).sum())
    y = rng.uniform(-0.28, 0.28, len(plac))
    ax.scatter(plac, y, s=14, color=PLAC, edgecolor="white", linewidth=0.5, zorder=2, label="Placebo graphs")
    ax.scatter([real], [0], s=70, marker="D", color=REAL, edgecolor="white", linewidth=1.2, zorder=4,
               label="Real relation graph")
    ax.scatter([none], [0], s=60, marker="o", color=NONE, edgecolor="white", linewidth=1.2, zorder=3,
               label="No cross-stock relations")
    ax.annotate(f"real: rank {rank} of {len(plac) + 1}, p = {rank / (len(plac) + 1):.2f}", (real, 0), xytext=(0, 17),
                textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    ax.set_ylim(-0.5, 1.0)
    ax.set_yticks([])
    ax.set_title(title, loc="left", pad=3)
    style(ax)
axes[-1].set_xlabel("Test-period IRR of the top-1 stock, seed 0 (common random numbers)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
fig.savefig(FIG / "fig_edge_placebos.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig_edge_placebos.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig. D1
rtres = json.load(open(RT / "results.json"))["descriptive"]
rows = []
for m, pub in (("NASDAQ", 1.19 - 0.68), ("NYSE", 0.79 - 0.56)):
    d = np.array([rsr_irr(m, "real", s) - rsr_irr(m, "none", s) for s in range(20)])
    rows.append((f"RSR, {m}", d, pub))
d = np.array(rtres["real_val_irr1"]) - np.array(rtres["none_val_irr1"])
rows.append(("RT-GCN(U), NASDAQ", d, 0.45 - 0.35))
ext = json.load(open(EXT / "results.json"))["cells"]["NASDAQ_T"]["descriptive"]
d = np.array(ext["real_val_irr1"]) - np.array(ext["none_val_irr1"])
rows.append(("RT-GCN(T), NASDAQ", d, 0.90))
PHOLM = {"RSR, NASDAQ": 0.217, "RSR, NYSE": 0.783, "RT-GCN(U), NASDAQ": 0.946, "RT-GCN(T), NASDAQ": 1.00}

fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.axvline(0, color=INK2, linewidth=0.8, zorder=1)
for i, (name, d, pub) in enumerate(rows):
    yy = len(rows) - 1 - i
    lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), d.std(ddof=1) / np.sqrt(len(d)))
    ax.scatter(d, yy + rng.uniform(-0.17, 0.17, len(d)), s=12, color=PLAC, edgecolor="white",
               linewidth=0.4, zorder=2, label="Single seed" if i == 0 else None)
    ax.plot([lo, hi], [yy - 0.32] * 2, color=INK, linewidth=2, solid_capstyle="round", zorder=3,
            label="Mean and 95% CI" if i == 0 else None)
    ax.scatter([d.mean()], [yy - 0.32], s=26, color=INK, zorder=4)
    ax.scatter([pub], [yy - 0.32], s=46, marker="D", facecolor="white", edgecolor=INK, linewidth=1.2,
               zorder=5, label="Published gain (vs. end-to-end Rank_LSTM)" if i == 0 else None)
    ax.annotate(f"n = {len(d)}\n$p_{{Holm}}$ = {PHOLM[name]:.2f}", (1.02, yy - 0.1), xycoords=("axes fraction", "data"),
                ha="left", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_xlabel("Test IRR (top-1) with relations minus without, per training seed")
style(ax)
ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.16))
fig.savefig(FIG / "fig_d1_seeds.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig_d1_seeds.png", dpi=200, bbox_inches="tight")
plt.close(fig)

for name, d, pub in rows:
    print(f"{name}: mean {d.mean():+.3f}, n_pos {(d > 0).sum()}/{len(d)}, published {pub:+.2f}")
for title, plac, real, none in panels:
    print(f"{title}: real {real:+.3f}, placebo mean {plac.mean():+.3f}, none {none:+.3f}, rank {1 + (plac >= real).sum()}")
