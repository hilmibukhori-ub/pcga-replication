"""Expert selectors. All selectors choose (or weight) between the same pair of frozen experts
(graph expert G, non-graph expert N) and, unless an ablation says otherwise, see the same
feature matrix. Fitting uses MetaData only (locked-test labels are absent by construction).

Hyper-parameters are selected on selector-val with each selector's own training objective;
affine calibration of benefit / risk predictions is fitted on the calibration split.
"""

from __future__ import annotations

import zlib

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn

from .features import GROUPS, MetaData

NON_STRUCT = ("evidence", "disagreement", "market")


# ----------------------------------------------------------------------------- helpers
class MetaMLP(nn.Module):
    def __init__(self, d: int, hidden: int, out: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


def _train_one(Xf, Xv, loss_fit, loss_val, out_dim, mcfg, wd, seed):
    torch.manual_seed(seed)
    m = MetaMLP(Xf.shape[1], mcfg["hidden"], out_dim, mcfg["dropout"])
    opt = torch.optim.Adam(m.parameters(), lr=mcfg["lr"], weight_decay=wd)
    best, state, bad = np.inf, None, 0
    for _ in range(mcfg["max_epochs"]):
        m.train()
        loss = loss_fit(m(Xf))
        opt.zero_grad()
        loss.backward()
        opt.step()
        m.eval()
        with torch.no_grad():
            v = float(loss_val(m(Xv)))
        if v < best - 1e-7:
            best, state, bad = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= mcfg["patience"]:
                break
    m.load_state_dict(state)
    m.eval()
    return m


class Ensemble:
    def __init__(self, models):
        self.models = models

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Xt = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return np.mean([m(Xt).numpy() for m in self.models], 0)


def fit_ensemble(Xf, Xv, loss_fit, loss_val, out_dim, mcfg, wd, seed):
    Xf_t = torch.as_tensor(Xf, dtype=torch.float32)
    Xv_t = torch.as_tensor(Xv, dtype=torch.float32)
    models = [_train_one(Xf_t, Xv_t, loss_fit, loss_val, out_dim, mcfg, wd, seed + k)
              for k in range(mcfg["n_init"])]
    ens = Ensemble(models)
    with torch.no_grad():
        v = float(loss_val(torch.as_tensor(ens(Xv), dtype=torch.float32)))
    return ens, v


def fit_affine(pred: np.ndarray, target: np.ndarray, mode: str):
    """target ~ a + c * pred. Falls back to an intercept shift if the slope is not positive."""
    if mode == "none":
        return 0.0, 1.0, "none"
    c, a = np.polyfit(pred, target, 1)
    if not np.isfinite(c) or c <= 0:
        return float(np.mean(target - pred)), 1.0, "shift_fallback"
    return float(a), float(c), "affine"


def _t(a):
    return torch.as_tensor(a, dtype=torch.float32)


def _finite_rows(X):
    return np.isfinite(X).all(1)


def _seed(base: int, name: str) -> int:
    return base * 1000 + zlib.crc32(name.encode()) % 997


def placebo_calibrated_weight(
    benefit: np.ndarray,
    placebo_gap: np.ndarray,
    benefit_scale: float,
    placebo_scale: float,
    temperature: float,
) -> np.ndarray:
    """Soft graph weight controlled by the weaker of no-graph and placebo evidence.

    Positive ``benefit`` says that the real-graph expert beats the matched non-graph
    expert (D1). Positive ``placebo_gap`` says that it beats retrained placebo-graph
    experts (D2). Taking the minimum makes either failed condition sufficient to
    suppress graph use. Scaling only makes the two evidence quantities comparable;
    it does not change either zero threshold.
    """
    if benefit_scale <= 0 or placebo_scale <= 0 or temperature <= 0:
        raise ValueError("PC-SGL scales and temperature must be positive")
    score = np.minimum(benefit / benefit_scale, placebo_gap / placebo_scale)
    z = np.clip(score / temperature, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------- selectors
class Selector:
    name = "selector"
    produces_benefit = False

    def fit(self, md: MetaData, seed: int, cfg: dict) -> dict:
        return {}

    def weight(self, md: MetaData) -> np.ndarray:
        raise NotImplementedError

    def benefit(self, md: MetaData):
        return None

    def d2(self, md: MetaData):
        return None


class Constant(Selector):
    def __init__(self, name, w):
        self.name, self.w = name, float(w)

    def weight(self, md):
        return np.full(md.Z.shape[0], self.w)


class FixedEnsemble(Selector):
    """Constant mixture weight fitted by least squares on meta-train."""

    name = "FixedEnsemble"

    def fit(self, md, seed, cfg):
        f = md.fit_idx
        d = md.pG[f] - md.pN[f]
        self.w = float(np.clip(((md.y[f] - md.pN[f]) * d).sum() / (d**2).sum(), 0, 1))
        return {"w": self.w}

    def weight(self, md):
        return np.full(md.Z.shape[0], self.w)


class BenefitSelector(Selector):
    """Predict the conditional benefit of the graph expert, E[L_N - L_G | z_t]; use G iff the
    calibrated prediction is positive. Optional auxiliary head predicts the realised D2."""

    produces_benefit = True

    def __init__(self, name, groups, model, aux_grid=(0.0,)):
        self.name, self.groups, self.model, self.aux_grid = name, tuple(groups), model, aux_grid

    def fit(self, md, seed, cfg):
        scfg = cfg["selectors"]
        self.cols = md.cols(self.groups)
        X = md.Z[:, self.cols]
        f, v, c = md.fit_idx, md.val_idx, md.cal_idx
        bm, bs = md.B[f].mean(), md.B[f].std()
        tb = (md.B - bm) / bs
        info = {"features": [md.names[i] for i in self.cols]}
        if self.model == "ridge":
            best = None
            for a in scfg["ridge_alpha_grid"]:
                r = Ridge(alpha=a).fit(X[f], tb[f])
                val = float(((r.predict(X[v]) - tb[v]) ** 2).mean())
                if best is None or val < best[0]:
                    best = (val, a, r)
            r = best[2]
            self._raw = lambda Xa: r.predict(Xa) * bs + bm
            self._raw_d2 = None
            info.update(alpha=best[1], val_mse_std=best[0])
        else:
            dm, ds = np.nanmean(md.D2[f]), np.nanstd(md.D2[f])
            td = (md.D2 - dm) / ds
            tbf, tbv, tdf = _t(tb[f]), _t(tb[v]), _t(td[f])
            best = None
            for wd in scfg["mlp"]["weight_decay_grid"]:
                for lam in self.aux_grid:
                    if lam > 0 and not np.isfinite(md.D2[f]).all():
                        continue
                    out_dim = 2 if lam > 0 else 1

                    def lf(o, lam=lam):
                        loss = ((o[:, 0] - tbf) ** 2).mean()
                        return loss + lam * ((o[:, 1] - tdf) ** 2).mean() if lam > 0 else loss

                    def lv(o):
                        return ((o[:, 0] - tbv) ** 2).mean()

                    ens, val = fit_ensemble(X[f], X[v], lf, lv, out_dim, scfg["mlp"], wd,
                                            _seed(seed, self.name))
                    if best is None or val < best[0]:
                        best = (val, wd, lam, ens)
            ens = best[3]
            self._raw = lambda Xa: ens(Xa)[:, 0] * bs + bm
            self._raw_d2 = (lambda Xa: ens(Xa)[:, 1] * ds + dm) if best[2] > 0 else None
            info.update(weight_decay=best[1], aux_weight=best[2], val_mse_std=best[0])
        self.a, self.c, mode = fit_affine(self._raw(X[c]), md.B[c], scfg["calibration"])
        info.update(calib_a=self.a, calib_c=self.c, calib_mode=mode)
        return info

    def _apply(self, md, fn):
        X = md.Z[:, self.cols]
        out = np.full(X.shape[0], np.nan)
        ok = _finite_rows(X)
        out[ok] = fn(X[ok])
        return out

    def benefit(self, md):
        return self._apply(md, lambda Xa: self.a + self.c * self._raw(Xa))

    def d2(self, md):
        return self._apply(md, self._raw_d2) if self._raw_d2 is not None else None

    def weight(self, md):
        b = self.benefit(md)
        return np.where(np.isfinite(b), (b > 0).astype(float), np.nan)


class DualEvidence(Selector):
    """A2b: separate structural-evidence and predictive-evidence benefit estimators; the graph
    expert is used only when both predict a positive benefit."""

    def __init__(self, name="A2b"):
        self.name = name
        self.s = BenefitSelector(name + "_struct", ("structural",), "mlp")
        self.e = BenefitSelector(name + "_evid", NON_STRUCT, "mlp")

    def fit(self, md, seed, cfg):
        return {"struct": self.s.fit(md, seed, cfg), "evidence": self.e.fit(md, seed, cfg)}

    def weight(self, md):
        bs, be = self.s.benefit(md), self.e.benefit(md)
        w = ((bs > 0) & (be > 0)).astype(float)
        return np.where(np.isfinite(bs) & np.isfinite(be), w, np.nan)


class CondRisk(Selector):
    """Predict each expert's conditional loss separately; choose the lower predicted risk."""

    produces_benefit = True

    def __init__(self, name="CondRisk", groups=GROUPS):
        self.name, self.groups = name, tuple(groups)

    def fit(self, md, seed, cfg):
        scfg = cfg["selectors"]
        self.cols = md.cols(self.groups)
        X = md.Z[:, self.cols]
        f, v, c = md.fit_idx, md.val_idx, md.cal_idx
        L = np.stack([md.LG, md.LN], 1)
        m, s = L[f].mean(0), L[f].std(0)
        tL = (L - m) / s
        tf, tv = _t(tL[f]), _t(tL[v])
        best = None
        for wd in scfg["mlp"]["weight_decay_grid"]:
            ens, val = fit_ensemble(X[f], X[v], lambda o: ((o - tf) ** 2).mean(),
                                    lambda o: ((o - tv) ** 2).mean(), 2, scfg["mlp"], wd,
                                    _seed(seed, self.name))
            if best is None or val < best[0]:
                best = (val, wd, ens)
        ens = best[2]
        self._raw = lambda Xa: ens(Xa) * s + m
        rc = self._raw(X[c])
        self.cal = [fit_affine(rc[:, k], L[c, k], scfg["calibration"]) for k in range(2)]
        return {"weight_decay": best[1], "val_mse_std": best[0],
                "calib": [list(x) for x in self.cal]}

    def _risks(self, md):
        X = md.Z[:, self.cols]
        out = np.full((X.shape[0], 2), np.nan)
        ok = _finite_rows(X)
        r = self._raw(X[ok])
        for k, (a, cc, _) in enumerate(self.cal):
            out[ok, k] = a + cc * r[:, k]
        return out

    def benefit(self, md):
        r = self._risks(md)
        return r[:, 1] - r[:, 0]

    def weight(self, md):
        b = self.benefit(md)
        return np.where(np.isfinite(b), (b > 0).astype(float), np.nan)


class Gate(Selector):
    """Generic learned soft gate w(z) in [0, 1] trained end-to-end on the mixture MSE."""

    def __init__(self, name="Gate", groups=GROUPS):
        self.name, self.groups = name, tuple(groups)

    def fit(self, md, seed, cfg):
        scfg = cfg["selectors"]
        self.cols = md.cols(self.groups)
        X = md.Z[:, self.cols]
        f, v = md.fit_idx, md.val_idx
        yf, gf, nf = _t(md.y[f]), _t(md.pG[f]), _t(md.pN[f])
        yv, gv, nv = _t(md.y[v]), _t(md.pG[v]), _t(md.pN[v])

        def mix_loss(o, y, g, n):
            w = torch.sigmoid(o[:, :1])
            return ((y - (w * g + (1 - w) * n)) ** 2).mean()

        best = None
        for wd in scfg["mlp"]["weight_decay_grid"]:
            ens, val = fit_ensemble(X[f], X[v], lambda o: mix_loss(o, yf, gf, nf),
                                    lambda o: mix_loss(o, yv, gv, nv), 1, scfg["mlp"], wd,
                                    _seed(seed, self.name))
            if best is None or val < best[0]:
                best = (val, wd, ens)
        self.ens = best[2]
        return {"weight_decay": best[1], "val_mix_mse": best[0]}

    def weight(self, md):
        X = md.Z[:, self.cols]
        out = np.full(X.shape[0], np.nan)
        ok = _finite_rows(X)
        out[ok] = 1 / (1 + np.exp(-self.ens(X[ok])[:, 0]))
        return out


class PlaceboCalibratedGate(Selector):
    """PC-SGL: use graph information only when both predicted D1 and D2 support it.

    Unlike A2a, whose D2 head is auxiliary and does not enter its decision rule,
    PC-SGL calibrates both heads and routes through their minimum standardized
    evidence. The temperature is chosen on a chronological calibration holdout;
    locked-test labels remain unavailable during fitting.
    """

    name = "PC-SGL"
    produces_benefit = True

    def __init__(self, groups=GROUPS):
        self.groups = tuple(groups)

    def fit(self, md, seed, cfg):
        scfg = cfg["selectors"]
        pcfg = scfg.get("pc_sgl", {})
        d2_weights = tuple(pcfg.get("d2_weight_grid", (0.5, 1.0, 2.0)))
        temperatures = tuple(pcfg.get("temperature_grid", (0.1, 0.25, 0.5, 1.0, 2.0)))
        self.cols = md.cols(self.groups)
        X = md.Z[:, self.cols]
        f, v, c = md.fit_idx, md.val_idx, md.cal_idx
        if not np.isfinite(md.D2[f]).all():
            raise ValueError("PC-SGL requires retrained placebo predictions on meta-train")

        self.b_mean, self.b_scale = float(md.B[f].mean()), float(md.B[f].std())
        self.d_mean, self.d_scale = float(md.D2[f].mean()), float(md.D2[f].std())
        self.b_scale = max(self.b_scale, 1e-8)
        self.d_scale = max(self.d_scale, 1e-8)
        tb = (md.B - self.b_mean) / self.b_scale
        td = (md.D2 - self.d_mean) / self.d_scale
        tbf, tbv, tdf, tdv = _t(tb[f]), _t(tb[v]), _t(td[f]), _t(td[v])

        best = None
        for wd in scfg["mlp"]["weight_decay_grid"]:
            for lam in d2_weights:
                def loss_fit(o, lam=lam):
                    return ((o[:, 0] - tbf) ** 2).mean() + lam * ((o[:, 1] - tdf) ** 2).mean()

                def loss_val(o):
                    return ((o[:, 0] - tbv) ** 2).mean() + ((o[:, 1] - tdv) ** 2).mean()

                ens, val = fit_ensemble(
                    X[f], X[v], loss_fit, loss_val, 2, scfg["mlp"], wd,
                    _seed(seed, f"{self.name}-{lam}"),
                )
                if best is None or val < best[0]:
                    best = (val, wd, lam, ens)
        self.ens = best[3]

        # Keep calibration and temperature selection chronological and disjoint.
        cut = len(c) // 2
        c_fit, c_tune = c[:cut], c[cut:]
        raw_c = self._raw(X[c_fit])
        self.b_cal = fit_affine(raw_c[:, 0], md.B[c_fit], scfg["calibration"])
        self.d_cal = fit_affine(raw_c[:, 1], md.D2[c_fit], scfg["calibration"])

        b_tune, d_tune = self._calibrated(X[c_tune])
        best_temp = None
        for temp in temperatures:
            w = placebo_calibrated_weight(
                b_tune, d_tune, self.b_scale, self.d_scale, float(temp)
            )
            pred = w[:, None] * md.pG[c_tune] + (1.0 - w[:, None]) * md.pN[c_tune]
            loss = float(((md.y[c_tune] - pred) ** 2).mean())
            if best_temp is None or loss < best_temp[0]:
                best_temp = (loss, float(temp))
        self.temperature = best_temp[1]
        return {
            "features": [md.names[i] for i in self.cols],
            "weight_decay": best[1],
            "d2_weight": best[2],
            "val_joint_mse_std": best[0],
            "benefit_calibration": list(self.b_cal),
            "d2_calibration": list(self.d_cal),
            "temperature": self.temperature,
            "calibration_mix_mse": best_temp[0],
            "calibration_fit_n": len(c_fit),
            "temperature_tune_n": len(c_tune),
        }

    def _raw(self, X):
        out = self.ens(X)
        return np.column_stack([
            out[:, 0] * self.b_scale + self.b_mean,
            out[:, 1] * self.d_scale + self.d_mean,
        ])

    def _calibrated(self, X):
        raw = self._raw(X)
        ab, cb, _ = self.b_cal
        ad, cd, _ = self.d_cal
        return ab + cb * raw[:, 0], ad + cd * raw[:, 1]

    def _predict(self, md):
        X = md.Z[:, self.cols]
        b = np.full(X.shape[0], np.nan)
        d = np.full(X.shape[0], np.nan)
        ok = _finite_rows(X)
        b[ok], d[ok] = self._calibrated(X[ok])
        return b, d

    def benefit(self, md):
        return self._predict(md)[0]

    def d2(self, md):
        return self._predict(md)[1]

    def weight(self, md):
        b, d = self._predict(md)
        out = np.full_like(b, np.nan)
        ok = np.isfinite(b) & np.isfinite(d)
        out[ok] = placebo_calibrated_weight(
            b[ok], d[ok], self.b_scale, self.d_scale, self.temperature
        )
        return out


def build_selectors(cfg: dict) -> list[Selector]:
    aux = tuple(cfg["selectors"]["aux_weight_grid"])
    return [
        Constant("AlwaysN", 0.0),
        Constant("AlwaysG", 1.0),
        FixedEnsemble(),
        Gate("Gate", GROUPS),
        CondRisk("CondRisk", GROUPS),
        BenefitSelector("A1", NON_STRUCT, "ridge"),
        BenefitSelector("A2a", GROUPS, "mlp", aux),
        BenefitSelector("A2a_noStruct", NON_STRUCT, "mlp", aux),
        BenefitSelector("A2a_noAux", GROUPS, "mlp", (0.0,)),
        DualEvidence("A2b"),
    ]
