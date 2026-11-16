"""D3 model comparison: next-year NPA count forecasts.

Primary target: published violent-crime summary, broad `Violent Crime` count per
NPA and calendar year (equals the named-offense sum; injury indicator excluded).
Sensitivity target: CMPD incident-report count per NPA and report year.

Design (fixed before any 2025 scoring):
  * Development data: target years <= 2024. Test: 2025, scored once. 2026 excluded.
  * Rolling-origin (expanding-window) validation inside development years.
      - tuning folds pick hyperparameters for each candidate;
      - later, untouched comparison folds compare baseline and candidates.
  * Same folds, same rows, same metrics for every model.
  * Selection rule: lowest mean comparison-fold MAE. If a more complex model is
    within one standard error of a simpler one, keep the simpler one
    (complexity order: baseline < Poisson GLM < random forest < boosting).
  * Features use only years strictly before the target year. NPA-years absent
    from the source are never filled with zero; a row needs the prior year
    observed, and the excluded rows are counted.

Run from the repository root:
    python d3_modeling/evaluate_models.py
Outputs go to d3_modeling/results/.
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dashboard" / "data"
OUT = Path(__file__).resolve().parent / "results"
SEED = 2301
TEST_YEAR = 2025

SOURCES = {
    "violent": dict(
        label="Violent-crime NPA summary (broad Violent Crime count)",
        tune_years=[2019, 2020, 2021],
        compare_years=[2022, 2023, 2024],
    ),
    "cmpd": dict(
        label="CMPD incident reports (all highest classifications)",
        tune_years=[2020, 2021],
        compare_years=[2022, 2023, 2024],
    ),
}

MODEL_ORDER = ["baseline_last_year", "poisson_glm", "random_forest", "gradient_boosting"]
MODEL_LABELS = {
    "baseline_last_year": "Baseline: last year's count",
    "poisson_glm": "Poisson GLM",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting (Poisson loss)",
}
GRIDS = {
    "poisson_glm": [dict(alpha=a) for a in (1e-4, 1e-2, 1e-1, 1.0)],
    "random_forest": [
        dict(min_samples_leaf=m, max_features=f) for m, f in product((5, 20), (0.5, 1.0))
    ],
    "gradient_boosting": [
        dict(learning_rate=lr, max_leaf_nodes=l, min_samples_leaf=20)
        for lr, l in product((0.03, 0.1), (7, 15))
    ],
}


# --------------------------------------------------------------------------- data
def load_series(source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (counts, extra) where counts has npa_id, year, y."""
    if source == "violent":
        v = pd.read_csv(DATA / "violent_npa_year_offense_counts.csv")
        broad = v[v.offense_level == "broad_category"]
        named = v[v.offense_level == "specific_offense"]
        y = broad.groupby(["npa_id", "year"]).offense_count.sum().rename("y")
        # Guard the D2 finding the target depends on.
        check = pd.concat([y, named.groupby(["npa_id", "year"]).offense_count.sum()], axis=1)
        assert (check.iloc[:, 0] == check.iloc[:, 1]).all(), "broad != named sum"
        d = named.assign(
            gun=named.offense_description.eq("Aggravated Assault-Gun") * named.offense_count,
            robbery=named.offense_description.str.contains("Robbery") * named.offense_count,
        ).groupby(["npa_id", "year"])[["gun", "robbery"]].sum()
        inj = (
            v[v.offense_level == "injury_indicator"]
            .groupby(["npa_id", "year"]).offense_count.sum().rename("injury")
        )
        extra = d.join(inj, how="left").reset_index()
        return y.reset_index(), extra
    c = pd.read_csv(DATA / "cmpd_npa_year_counts.csv").rename(columns={"report_count": "y"})
    return c[["npa_id", "year", "y"]], pd.DataFrame(columns=["npa_id", "year"])


def build_features(counts: pd.DataFrame, extra: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    counts = counts[counts.year <= TEST_YEAR]
    wide = counts.pivot(index="npa_id", columns="year", values="y")
    city = counts.groupby("year").y.sum()
    ex = extra.set_index(["npa_id", "year"]) if len(extra) else None
    rows, excluded = [], {}
    years = sorted(wide.columns)
    for t in years[1:]:
        for npa, y in wide[t].dropna().items():
            l1 = wide.at[npa, t - 1] if t - 1 in wide.columns else np.nan
            if np.isnan(l1):
                excluded[t] = excluded.get(t, 0) + 1
                continue
            l2 = wide.at[npa, t - 2] if t - 2 in wide.columns else np.nan
            l3 = wide.at[npa, t - 3] if t - 3 in wide.columns else np.nan
            hist = wide.loc[npa, [c for c in years if c < t]]
            r = dict(
                npa_id=npa, year=t, y=y, lag1=l1,
                lag2_missing=float(np.isnan(l2)), lag3_missing=float(np.isnan(l3)),
                lag2=l1 if np.isnan(l2) else l2, lag3=(l1 if np.isnan(l2) else l2) if np.isnan(l3) else l3,
                years_observed=float(hist.notna().sum()),
                hist_max=float(hist.max()),
                city_lag1=float(city.get(t - 1)),
                city_change=float(city.get(t - 1) / city.get(t - 2)) if (t - 2) in city.index else 1.0,
            )
            if ex is not None:
                e = ex.loc[(npa, t - 1)] if (npa, t - 1) in ex.index else None
                r["gun_share_lag1"] = float(e["gun"] / l1) if e is not None else 0.0
                r["robbery_share_lag1"] = float(e["robbery"] / l1) if e is not None else 0.0
                inj = e["injury"] if e is not None else np.nan
                # Absent injury rows are unknown, not zero: keep an explicit flag.
                r["injury_lag1_missing"] = float(pd.isna(inj))
                r["injury_lag1"] = 0.0 if pd.isna(inj) else float(inj)
            rows.append(r)
    df = pd.DataFrame(rows)
    df["mean3"] = df[["lag1", "lag2", "lag3"]].mean(axis=1)
    df["trend"] = df.lag1 - df.lag3
    for col in ["lag1", "lag2", "lag3", "mean3", "hist_max"]:
        df[f"log_{col}"] = np.log1p(df[col])
    df["log_city_lag1"] = np.log(df.city_lag1)
    return df, excluded


def feature_cols(df: pd.DataFrame) -> list[str]:
    base = ["log_lag1", "log_lag2", "log_lag3", "log_mean3", "log_hist_max", "trend",
            "lag2_missing", "lag3_missing", "years_observed", "log_city_lag1", "city_change"]
    extra = ["gun_share_lag1", "robbery_share_lag1", "injury_lag1", "injury_lag1_missing"]
    return base + [c for c in extra if c in df.columns]


# ------------------------------------------------------------------------- models
def make_model(name: str, params: dict):
    if name == "poisson_glm":
        return make_pipeline(StandardScaler(), PoissonRegressor(max_iter=3000, **params))
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1, **params)
    if name == "gradient_boosting":
        return HistGradientBoostingRegressor(loss="poisson", max_iter=400, early_stopping=False,
                                             random_state=SEED, **params)
    raise ValueError(name)


def fit_predict(name, params, train, test, cols):
    if name == "baseline_last_year":
        return test.lag1.to_numpy(dtype=float)
    m = make_model(name, params)
    if name == "random_forest":  # fit on log scale, back-transform
        m.fit(train[cols], np.log1p(train.y))
        return np.clip(np.expm1(m.predict(test[cols])), 0.1, None)
    m.fit(train[cols], train.y)
    return np.clip(m.predict(test[cols]), 0.1, None)


# ------------------------------------------------------------------------ metrics
def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    p = np.clip(p, 0.1, None)
    dev = 2 * np.mean(np.where(y > 0, y * np.log(y / p), 0) - (y - p))
    k = max(1, int(round(0.1 * len(y))))
    top_true, top_pred = set(np.argsort(-y)[:k]), set(np.argsort(-p)[:k])
    return dict(
        n=len(y),
        mae=float(np.mean(np.abs(y - p))),
        rmse=float(np.sqrt(np.mean((y - p) ** 2))),
        poisson_deviance=float(dev),
        bias=float(np.mean(p - y)),
        spearman=float(spearmanr(y, p).statistic),
        top10_overlap=len(top_true & top_pred) / k,
    )


def rolling_cv(df, name, params, years, cols):
    preds = []
    for v in years:
        tr, te = df[df.year < v], df[df.year == v]
        preds.append(te.assign(pred=fit_predict(name, params, tr, te, cols), model=name))
    return pd.concat(preds)


def evaluate_source(source: str, cfg: dict) -> dict:
    counts, extra = load_series(source)
    df, excluded = build_features(counts, extra)
    cols = feature_cols(df)
    dev = df[df.year < TEST_YEAR]
    assert dev.year.max() == TEST_YEAR - 1 and TEST_YEAR not in dev.year.values

    # 1) tune each candidate on tuning folds (development years only)
    tuned, tuning_log = {"baseline_last_year": {}}, []
    for name, grid in GRIDS.items():
        best = None
        for params in grid:
            cv = rolling_cv(dev, name, params, cfg["tune_years"], cols)
            mae = cv.assign(ae=(cv.y - cv.pred).abs()).groupby("year").ae.mean().mean()
            tuning_log.append(dict(model=name, params=json.dumps(params), tune_mae=mae))
            if best is None or mae < best[0]:
                best = (mae, params)
        tuned[name] = best[1]

    # 2) compare on later, untouched development folds
    cv_preds = pd.concat(
        rolling_cv(dev, n, tuned[n], cfg["compare_years"], cols) for n in MODEL_ORDER
    )
    fold = (
        cv_preds.groupby(["model", "year"])
        .apply(lambda g: pd.Series(metrics(g.y, g.pred)), include_groups=False)
        .reset_index()
    )
    summary = fold.groupby("model").agg(
        **{f"{m}_mean": (m, "mean") for m in ["mae", "rmse", "poisson_deviance", "bias",
                                                  "spearman", "top10_overlap"]},
        mae_se=("mae", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
    ).reindex(MODEL_ORDER)
    base_mae = fold[fold.model == "baseline_last_year"].set_index("year").mae
    summary["folds_beating_baseline"] = [
        int((fold[fold.model == m].set_index("year").mae < base_mae).sum()) for m in MODEL_ORDER
    ]

    # 3) selection rule (CV only)
    best = summary.mae_mean.idxmin()
    selected = best
    for m in MODEL_ORDER:  # simplest model within 1 SE of the best
        if summary.at[m, "mae_mean"] <= summary.at[best, "mae_mean"] + summary.at[best, "mae_se"]:
            selected = m
            break

    # 4) single 2025 test: refit every model on all development rows
    test = df[df.year == TEST_YEAR]
    test_preds = pd.concat(
        test.assign(pred=fit_predict(n, tuned[n], dev, test, cols), model=n) for n in MODEL_ORDER
    )
    test_metrics = (
        test_preds.groupby("model").apply(lambda g: pd.Series(metrics(g.y, g.pred)), include_groups=False)
        .reindex(MODEL_ORDER)
    )
    # Diagnostic only (uses the 2025 total, so it is NOT a valid forecast):
    # how much error is the citywide level shift versus between-NPA allocation?
    level_adj = {}
    for n, g in test_preds.groupby("model"):
        level_adj[n] = float(np.mean(np.abs(g.y - g.pred * g.y.sum() / g.pred.sum())))
    test_metrics["mae_level_adjusted_diagnostic"] = pd.Series(level_adj)

    # 5) error analysis for the selected model (plus baseline for contrast)
    tiers = pd.qcut(dev.lag1, 5, retbins=True, duplicates="drop")[1]
    tiers[0], tiers[-1] = -np.inf, np.inf
    tier_labels = [f"Q{i+1}" for i in range(len(tiers) - 1)]

    def groups(frame):
        g = frame.copy()
        g["prior_volume_tier"] = pd.cut(g.lag1, tiers, labels=tier_labels)
        chg = (g.y - g.lag1) / g.lag1
        g["actual_change"] = pd.cut(chg, [-np.inf, -0.25, 0.25, np.inf],
                                    labels=["fell >25%", "within ±25%", "rose >25%"])
        g["history"] = np.where(g.lag3_missing + g.lag2_missing > 0, "short (<3 prior yrs)", "full 3-yr")
        g["err"] = g.pred - g.y
        g["abs_err"] = g.err.abs()
        g["pct_err"] = g.err / g.y
        return g

    err_rows = []
    for split, frame in [("cv_2022_2024", cv_preds), ("test_2025", test_preds)]:
        g = groups(frame[frame.model.isin([selected, "baseline_last_year"])])
        for gvar in ["prior_volume_tier", "actual_change", "history"]:
            t = g.groupby(["model", gvar], observed=True).agg(
                n=("y", "size"), mean_actual=("y", "mean"), mean_pred=("pred", "mean"),
                mae=("abs_err", "mean"), bias=("err", "mean"),
                median_pct_err=("pct_err", "median"),
            ).reset_index().rename(columns={gvar: "group"})
            t.insert(0, "grouping", gvar)
            t.insert(0, "split", split)
            err_rows.append(t)
        if split == "cv_2022_2024":
            t = g.groupby(["model", "year"]).agg(
                n=("y", "size"), mean_actual=("y", "mean"), mean_pred=("pred", "mean"),
                mae=("abs_err", "mean"), bias=("err", "mean"),
                median_pct_err=("pct_err", "median"),
            ).reset_index().rename(columns={"year": "group"})
            t.insert(0, "grouping", "fold_year")
            t.insert(0, "split", split)
            err_rows.append(t)
    errors = pd.concat(err_rows, ignore_index=True)

    # Top-decile ("highest-count") membership: who would a score flag?
    sel = test_preds[test_preds.model == selected].copy()
    k = int(round(0.1 * len(sel)))
    sel["actual_top"] = sel.y.rank(ascending=False, method="first") <= k
    sel["pred_top"] = sel.pred.rank(ascending=False, method="first") <= k
    top_table = pd.crosstab(sel.actual_top, sel.pred_top).to_dict()
    largest = groups(sel).sort_values("abs_err", ascending=False).head(10)[
        ["npa_id", "lag1", "y", "pred", "err", "prior_volume_tier", "actual_change"]]

    # ------------------------------------------------------------------ write
    out = OUT / source
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tuning_log).to_csv(out / "tuning_log.csv", index=False)
    fold.to_csv(out / "cv_fold_metrics.csv", index=False)
    summary.to_csv(out / "cv_summary.csv")
    test_metrics.to_csv(out / "test_2025_metrics.csv")
    errors.to_csv(out / "error_analysis.csv", index=False)
    largest.to_csv(out / "test_2025_largest_errors.csv", index=False)
    pd.concat([cv_preds.assign(split="cv"), test_preds.assign(split="test")])[
        ["split", "model", "npa_id", "year", "y", "lag1", "pred"]
    ].to_csv(out / "predictions.csv", index=False)
    info = dict(
        source=cfg["label"], selected_model=selected, best_cv_mae_model=best,
        tuned_params=tuned, features=cols,
        rows_total=int(len(df)), rows_dev=int(len(dev)), rows_test=int(len(test)),
        excluded_no_prior_year=excluded,
        npas_in_test=int(test.npa_id.nunique()),
        city_total_2024=int(counts[counts.year == 2024].y.sum()),
        city_total_2025=int(counts[counts.year == 2025].y.sum()),
        top_decile_k=k, top_decile_crosstab=json.loads(json.dumps(top_table, default=str)),
        tier_cutpoints_lag1=[float(x) for x in tiers[1:-1]],
    )
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    return dict(info=info, summary=summary, fold=fold, test=test_metrics, errors=errors)


# -------------------------------------------------------------------------- plots
def plots(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline_last_year": "#8a8984", "poisson_glm": "#2a78d6",
              "random_forest": "#eb6834", "gradient_boosting": "#1baf7a"}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})

    # Fig 1: MAE by validation fold + 2025 test, violent summary
    r = res["violent"]
    fold = r["fold"]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    years = sorted(fold.year.unique()) + [TEST_YEAR]
    x = np.arange(len(years))
    w = 0.2
    for i, m in enumerate(MODEL_ORDER):
        vals = list(fold[fold.model == m].set_index("year").mae.reindex(years[:-1])) + [
            r["test"].at[m, "mae"]]
        ax.bar(x + (i - 1.5) * w, vals, w * 0.9, color=colors[m], label=MODEL_LABELS[m])
    ax.axvline(len(years) - 1.5, color="#c3c2b7", lw=1, ls="--")
    ax.set_xticks(x, [str(y) for y in years[:-1]] + ["2025\n(held-out test)"])
    ax.set_ylabel("MAE (violent offenses per NPA)")
    ax.yaxis.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 13)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
    ax.set_title("Mean absolute error by validation year and on the 2025 test",
                 fontsize=10, loc="left", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_mae_by_year.png", dpi=200)
    plt.close(fig)

    # Fig 2: 2025 bias by prior-volume tier, selected vs baseline
    sel = r["info"]["selected_model"]
    e = r["errors"]
    e = e[(e.split == "test_2025") & (e.grouping == "prior_volume_tier")]
    cuts = r["info"]["tier_cutpoints_lag1"]
    edges = [0] + cuts
    tlabels = [f"{lab}\n(≤{int(c)})" if j < len(cuts) else f"{lab}\n(>{int(cuts[-1])})"
               for j, (lab, c) in enumerate(zip(sorted(e.group.unique()), cuts + [None]))]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, metric, lab in [(axes[0], "bias", "Mean error (pred − actual)"),
                            (axes[1], "median_pct_err", "Median % error vs. actual")]:
        for i, m in enumerate(["baseline_last_year", sel]):
            d = e[e.model == m].set_index("group")[metric]
            vals = d.values * (100 if metric == "median_pct_err" else 1)
            ax.bar(np.arange(len(d)) + (i - 0.5) * 0.38, vals, 0.34, color=colors[m],
                   label=MODEL_LABELS[m])
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_xticks(np.arange(len(tlabels)), tlabels, fontsize=7.5)
        ax.set_ylabel(lab)
        ax.yaxis.grid(True, color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("2025 test errors by prior-year volume quintile (cut points from development years)", fontsize=10,
                 x=0.02, ha="left", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_test_error_by_tier.png", dpi=200)
    plt.close(fig)


def main() -> None:
    res = {s: evaluate_source(s, c) for s, c in SOURCES.items()}
    plots(res)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    for s, r in res.items():
        print(f"\n===== {r['info']['source']}")
        print("selected:", r["info"]["selected_model"], "| tuned:", r["info"]["tuned_params"])
        print("rows dev/test:", r["info"]["rows_dev"], r["info"]["rows_test"],
              "| excluded (no prior year):", r["info"]["excluded_no_prior_year"])
        print("\nCV comparison folds (mean over years):\n", r["summary"].round(3))
        print("\nFold metrics:\n", r["fold"][["model", "year", "mae", "bias", "spearman"]].round(3))
        print("\n2025 test:\n", r["test"].round(3))
        print("\nErrors:\n", r["errors"].round(2).to_string())
        print("\ninfo:", json.dumps({k: r["info"][k] for k in
              ["city_total_2024", "city_total_2025", "top_decile_k", "top_decile_crosstab",
               "tier_cutpoints_lag1"]}, default=str))


if __name__ == "__main__":
    main()
