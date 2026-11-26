"""Five-part validation framework (parts 1b, 2, 3 here; 1a/4/5 come from evaluate.py results).

1b Cross-year proxy: for a window in year Y, compare the model's NORMAL-scenario forecast CHANGE
   (forecast - last observed context NDVI) with the observed change at the same location, same bins,
   in other years whose horizon was NOT a drought. Other years differ in other ways too, so this is a
   proxy, not truth.
2  Natural experiment: drought-labelled windows (treated) at test locations are matched to
   non-drought windows (controls) in the same year and bins at other held-out locations (val/test
   locations, never used for training) on static attributes, latitude and the pre-forecast NDVI
   trajectory. Covariate scaling is fitted on TRAIN locations/years only. Observed effect:
   DiD = (treated_post - treated_pre) - (control_post - control_pre). Compared with the model's
   estimated effect (observed-weather forecast minus normal-scenario forecast) for the treated unit.
3  Placebos: (a) scenario/noise placebos (two different normal analog sets with CRN; same normal
   weather with independent noise), (b) temporal placebo AUC, (c) permutation of drought labels.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..data.stress import is_drought

PLACEBO_TOL = 0.01   # NDVI


def _nanmean(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def window_is_drought(cube, cfg, l: int, yi: int, t0: int) -> bool:
    K, H = cfg["window"]["context"], cfg["window"]["horizon"]
    bd = cube.meta["bin_days"]
    return is_drought(cube.s_daily[l, yi, (t0 + K) * bd:(t0 + K + H) * bd], cfg["drought"]["threshold"],
                      cfg["drought"]["min_stressed_days"])


def observed_change(cube, cfg, l: int, yi: int, t0: int, oy: int, ox: int) -> np.ndarray:
    """Observed NDVI change from the last valid context value (per pixel) over the horizon. [H,P,P]."""
    K, H, P = cfg["window"]["context"], cfg["window"]["horizon"], cfg["window"]["patch"]
    seq = np.asarray(cube.ndvi[l, yi, t0:t0 + K + H, oy:oy + P, ox:ox + P], dtype=np.float32)
    last = np.full(seq.shape[1:], np.nan, dtype=np.float32)
    for t in range(K):
        ok = np.isfinite(seq[t])
        last[ok] = seq[t][ok]
    return seq[K:] - last[None]


def cross_year_proxy(cube, cfg, l: int, yi: int, t0: int, oy: int, ox: int, fc_change: dict,
                     model_anom: np.ndarray | None = None) -> dict:
    """fc_change: name -> forecast change [H,P,P] relative to the forecast start (last_ndvi).
    Returns MAE of each forecast change against the mean observed change in non-drought other years.
    With model_anom ([H,P,P] = observed-weather forecast minus normal forecast), also returns the
    OBSERVED anomaly of this year (observed change minus the reference change) and the model anomaly
    on the same pixels (observation-based agreement check, red-team C1)."""
    others = [y for y in range(len(cube.years)) if y != yi and not window_is_drought(cube, cfg, l, y, t0)]
    if not others:
        return {"xp_n_years": 0}
    ref = _nanmean(np.stack([observed_change(cube, cfg, l, y, t0, oy, ox) for y in others]), axis=0)
    ok = np.isfinite(ref)
    out = {"xp_n_years": len(others), "xp_n_pix": int(ok.sum())}
    if ok.sum() == 0:
        return out
    for k, v in fc_change.items():
        out[f"xp_mae_{k}"] = float(np.abs(v[ok] - ref[ok]).mean())
    out["xp_ref_mean_change"] = float(ref[ok].mean())
    if model_anom is not None:
        this = observed_change(cube, cfg, l, yi, t0, oy, ox)
        ok2 = ok & np.isfinite(this)
        out["xp_n_pix_anom"] = int(ok2.sum())
        if ok2.sum() > 0:
            out["xp_obs_anom"] = float((this[ok2] - ref[ok2]).mean())
            out["xp_model_anom"] = float(model_anom[ok2].mean())
    return out


# ----------------------------------------------------------------------------- natural experiment
def unit_features(cube, cfg, l: int, yi: int, t0: int, oy: int, ox: int, P: int | None = None) -> dict:
    K, H = cfg["window"]["context"], cfg["window"]["horizon"]
    P = P or cfg["window"]["patch"]
    seq = np.asarray(cube.ndvi[l, yi, t0:t0 + K + H, oy:oy + P, ox:ox + P], dtype=np.float32)
    fm = _nanmean(seq.reshape(K + H, -1), axis=1)            # frame means (covariates only)
    lc = cube.lcg[l, oy:oy + P, ox:ox + P]
    pre_pix, post_pix, msk = did_pixels(seq, K)
    return {
        # pre/post on a COMMON pixel set (pixels observed both before and after the forecast start)
        "pre": float(pre_pix[msk].mean()) if msk.any() else np.nan,
        "post": float(post_pix[msk].mean()) if msk.any() else np.nan,
        "ctx_level": _nanmean(fm[:K]), "ctx_trend": _nanmean(fm[K - 3:K]) - _nanmean(fm[:3]),
        "elev": float(np.nanmean(cube.static[l, 0, oy:oy + P, ox:ox + P])), "lat": float(cube.meta["lat"][l]),
        "f_forest": float((lc == 0).mean()), "f_grass": float((lc == 1).mean()), "f_crop": float((lc == 2).mean()),
    }


def did_pixels(seq: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel pre (mean of last 2 context bins) and post (horizon mean), and the common valid mask."""
    pre = _nanmean(seq[K - 2:K], axis=0)
    post = _nanmean(seq[K:], axis=0)
    return pre, post, np.isfinite(pre) & np.isfinite(post)


COVS = ["ctx_level", "ctx_trend", "elev", "lat", "f_forest", "f_grass", "f_crop"]


def covariate_scale(cube, cfg) -> tuple[pd.Series, pd.Series]:
    """Mean/std of matching covariates over TRAIN windows (train locations x train years)."""
    from ..data.dataset import make_windows

    w = make_windows(cube, cfg, "train", stride=4, quadrants=True)   # same unit size as matching
    df = pd.DataFrame([unit_features(cube, cfg, *map(int, r)) for r in w])
    return df[COVS].mean(), df[COVS].std().replace(0, 1.0)


def smd(t: pd.DataFrame, c: pd.DataFrame) -> pd.Series:
    s = np.sqrt((t[COVS].var() + c[COVS].var()) / 2).replace(0, np.nan)
    return ((t[COVS].mean() - c[COVS].mean()) / s).fillna(0.0)


def natural_experiment(cube, cfg, effects: pd.DataFrame, caliper: float = 0.5, n_controls: int = 3) -> dict:
    """effects: table from scripts/effects.py (test windows). Returns dict of DataFrames/metrics.
    Controls must have near-typical weather (ScenarioBuilder.is_normal), not merely 'not drought'.
    The model effect is averaged over the same pixels as the DiD (column dON_did_*), if available."""
    from ..data.scenarios import ScenarioBuilder

    sb = ScenarioBuilder(cube, cfg)
    P = cfg["window"]["patch"]
    mu, sd = covariate_scale(cube, cfg)
    yrs = list(cube.years)
    treated = effects[effects.drought == 1.0].copy()
    pool_locs = np.nonzero(np.isin(cube.split, ["val", "test"]))[0]
    full = cube.ndvi.shape[-1]
    t_rows, c_rows, pairs = [], [], []
    for r in treated.itertuples():
        yi = yrs.index(int(r.year))
        tf = unit_features(cube, cfg, int(r.loc), yi, int(r.t0), int(r.oy), int(r.ox))
        if not np.isfinite([tf["pre"], tf["post"], tf["ctx_level"]]).all():
            continue
        cands = []
        for cl in pool_locs:
            if cl == r.loc or not sb.is_normal(cl, yi, int(r.t0)):
                continue
            for oy in range(0, full - P + 1, P):
                for ox in range(0, full - P + 1, P):
                    cf = unit_features(cube, cfg, cl, yi, int(r.t0), oy, ox)
                    if np.isfinite([cf["pre"], cf["post"], cf["ctx_level"], cf["ctx_trend"]]).all():
                        cands.append(cf)
        if not cands:
            continue
        cdf = pd.DataFrame(cands)
        z_t = (pd.Series({k: tf[k] for k in COVS}) - mu) / sd
        z_c = (cdf[COVS] - mu) / sd
        dist = np.sqrt(((z_c - z_t) ** 2).mean(axis=1))
        order = np.argsort(dist.values)[:n_controls]
        good = [i for i in order if dist.values[i] <= caliper]
        if not good:
            continue
        ctrl = cdf.iloc[good]
        did = (tf["post"] - tf["pre"]) - float((ctrl.post - ctrl.pre).mean())
        t_rows.append(tf)
        c_rows.extend(ctrl.to_dict("records"))
        pre_ = "dON_did" if hasattr(r, "dON_did_mean") and np.isfinite(r.dON_did_mean) else "dON"
        pairs.append({"loc": r.loc, "year": r.year, "t0": r.t0, "did": did, "model_mean": getattr(r, f"{pre_}_mean"),
                      "model_q05": getattr(r, f"{pre_}_q05"), "model_q95": getattr(r, f"{pre_}_q95"),
                      "model_pixels": pre_, "n_controls": len(good),
                      "match_dist": float(dist.values[good].mean())})
    pairs_df = pd.DataFrame(pairs)
    res = {"n_treated_total": int(len(treated)), "n_matched": int(len(pairs_df))}
    if len(pairs_df) >= 3:
        t_df, c_df = pd.DataFrame(t_rows), pd.DataFrame(c_rows)
        bal = smd(t_df, c_df)
        res["max_abs_smd"] = float(bal.abs().max())
        res["smd"] = bal.to_dict()
        res["did_mean"] = float(pairs_df.did.mean())
        res["model_mean"] = float(pairs_df.model_mean.mean())
        res["bias_model_minus_did"] = float((pairs_df.model_mean - pairs_df.did).mean())
        res["corr"] = float(np.corrcoef(pairs_df.model_mean, pairs_df.did)[0, 1]) if pairs_df.did.std() > 0 else np.nan
        inside = (pairs_df.did >= pairs_df.model_q05) & (pairs_df.did <= pairs_df.model_q95)
        res["did_in_model_90pi"] = float(inside.mean())
    return {"summary": res, "pairs": pairs_df}


# ----------------------------------------------------------------------------- placebos
def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) with ties counted half (Mann-Whitney)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().values
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def placebos(effects: pd.DataFrame, n_perm: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for k, name in [("dN2", "scenario_placebo"), ("dNi", "noise_placebo")]:
        # scenario placebo is only informative when >= 2 distinct normal analogs exist (else N2 == N
        # identically and the test could not fail) -> restrict and report the count
        e_k = effects[effects.n_analogs >= 2] if k == "dN2" else effects
        out[f"{name}_n"] = int(len(e_k))
        if not len(e_k):
            continue
        v = e_k[f"{k}_mean"].values
        out[f"{name}_mean"] = float(v.mean())
        out[f"{name}_abs_mean"] = float(np.abs(v).mean())
        boot = [rng.choice(v, len(v)).mean() for _ in range(1000)]
        out[f"{name}_lo95"], out[f"{name}_hi95"] = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        out[f"{name}_frac_below_tol"] = float((np.abs(v) < PLACEBO_TOL).mean())
        # does the member interval include 0?
        out[f"{name}_frac_pi_covers_0"] = float(((e_k[f"{k}_q05"] <= 0) & (e_k[f"{k}_q95"] >= 0)).mean())
    out["drought_effect_abs_mean"] = float(np.abs(effects.loc[effects.drought == 1, "dON_mean"]).mean()) \
        if (effects.drought == 1).any() else float("nan")
    d = effects.drought.values == 1
    score = -effects.dON_mean.values                       # larger loss => more drought-like
    out["temporal_auc"] = auc(score[d], score[~d])
    out["n_drought"], out["n_normal"] = int(d.sum()), int((~d).sum())
    out["dON_mean_normal_windows"] = float(effects.dON_mean.values[~d].mean()) if (~d).any() else float("nan")
    out["dON_mean_drought_windows"] = float(effects.dON_mean.values[d].mean()) if d.any() else float("nan")
    # permutation placebo: shuffle drought labels ACROSS LOCATIONS (location-level label vectors)
    contrast = lambda lab: effects.dON_mean.values[lab].mean() - effects.dON_mean.values[~lab].mean()  # noqa: E731
    if d.any() and (~d).any():
        real = contrast(d)
        locs = effects["loc"].values
        uniq = np.unique(locs)
        perm = []
        for _ in range(n_perm):
            mapping = dict(zip(uniq, rng.permutation(uniq)))
            # each window takes the label of the same (year, t0) window at the mapped location, if present
            key = {(lo, y, t): dr for lo, y, t, dr in zip(effects["loc"], effects.year, effects.t0, d)}
            lab = np.array([key.get((mapping[lo], y, t), rng.random() < d.mean())
                            for lo, y, t in zip(effects["loc"], effects.year, effects.t0)])
            if lab.any() and (~lab).any():
                perm.append(contrast(lab))
        perm = np.array(perm)
        out["perm_real_contrast"] = float(real)
        out["perm_null_mean"] = float(perm.mean())
        out["perm_null_sd"] = float(perm.std())
        out["perm_p_one_sided"] = float((np.sum(perm <= real) + 1) / (len(perm) + 1))
    out["crn_var_ratio"] = float(effects.var_crn.mean() / effects.var_ind.mean()) if effects.var_ind.mean() > 0 \
        else float("nan")
    return out


def anomaly_agreement(effects: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """Observation-based check (red-team C1): does the model's weather-driven anomaly
    (observed-weather forecast minus normal forecast) track the OBSERVED anomaly (observed change minus
    the mean change in non-drought other years at the same place and bins)? Correlation and slope with
    95 % bootstrap CIs over locations."""
    d = effects.dropna(subset=["xp_obs_anom", "xp_model_anom"]) if "xp_obs_anom" in effects else effects.iloc[:0]
    out = {"anom_n": int(len(d))}
    if len(d) < 10 or d.xp_model_anom.std() == 0:
        return out
    rng = np.random.default_rng(seed)
    locs = d["loc"].unique()

    def stats(df):
        c = np.corrcoef(df.xp_model_anom, df.xp_obs_anom)[0, 1]
        slope = np.polyfit(df.xp_model_anom, df.xp_obs_anom, 1)[0]
        return c, slope

    c, sl = stats(d)
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(locs, len(locs))
        dd = pd.concat([d[d["loc"] == lo] for lo in pick])
        if dd.xp_model_anom.std() > 0:
            bs.append(stats(dd))
    bs = np.array(bs)
    out.update({"anom_corr": float(c), "anom_corr_lo95": float(np.percentile(bs[:, 0], 2.5)),
                "anom_corr_hi95": float(np.percentile(bs[:, 0], 97.5)), "anom_slope": float(sl),
                "anom_mean_obs_drought": float(d.loc[d.drought == 1, "xp_obs_anom"].mean()) if (d.drought == 1).any() else np.nan,
                "anom_mean_model_drought": float(d.loc[d.drought == 1, "xp_model_anom"].mean()) if (d.drought == 1).any() else np.nan})
    return out
