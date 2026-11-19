"""Descriptive data check (post hoc, requested in review): what is the +3,321% NYSE stock-day on test day 89?
Scans the released RT-GCN price arrays for split-like jumps (day-on-day price ratio > 5 or < 0.2) and keeps
those whose 10-day median level after the jump differs from the 10-day level before it by > 4x (persistent).
The released data contain no tickers or dates, so a corporate action cannot be confirmed by name.
Labels are next-day returns: label[t] = price[t+1] / price[t] - 1 (verified below)."""
import json, pathlib, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; D = ROOT / "data/raw/rtgcn_dl"
res = {}
for m in ("NASDAQ", "NYSE"):
    Ptr, P, L = (np.load(D / m / f) for f in ("train_15_price.npy", "test_15_price.npy", "test_15_label.npy"))
    ok = np.allclose(L[:-1], np.where(P[:-1] > 0, P[1:] / np.where(P[:-1] > 0, P[:-1], 1) - 1, L[:-1]), atol=1e-4)
    R = P[1:] / np.where(P[:-1] > 0, P[:-1], np.nan)
    ev = []
    for t, i in zip(*np.where((R > 5) | (R < 0.2))):
        t1 = t + 1
        if 10 <= t1 <= len(P) - 10:
            pre, post = np.nanmedian(P[t1 - 10:t1, i]), np.nanmedian(P[t1:t1 + 10, i])
            ev.append(dict(test_day_of_label=int(t), stock=int(i), jump=float(R[t, i]), persistent_ratio=float(post / pre),
                           label=float(L[t, i]), level_before_vs_sample_max=float(pre / np.concatenate([Ptr[:, i], P[:, i]]).max())))
    top = np.sort(L.ravel())[::-1][:3].round(3).tolist()
    res[m] = dict(label_is_next_day_return=bool(ok), split_like_events_in_test=ev, top3_labels=top)
    print(m, "label=next-day return:", ok, "| top labels", top, "| events", ev)
(ROOT / "outputs/rtgcn_ext_v1").mkdir(parents=True, exist_ok=True)
json.dump(res, open(ROOT / "outputs/rtgcn_ext_v1/day89_data_check.json", "w"), indent=1)
