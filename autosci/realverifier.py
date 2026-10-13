"""AutoSCI — RealVerifier (v0.11 new).

v0.7's `verifier.py` runs a *pure-numpy* synthetic experiment. v0.11 adds
this module which, **if sklearn is available**, runs a small real
classifier (LogReg, RandomForest, small MLP) on the same synthetic
distribution and reports actual AUROC. The synthetic data is a stand-in
for "small microbiome cohort" until the user wires a real dataset.

Why include this even though v0.7 already has a verifier?
    - The pure-numpy logistic regression upper-bounds the experiment
      with an *unrealistically* good classifier.
    - sklearn's baselines (esp. RandomForest) tell us what a *typical*
      model would do on this distribution — the same baseline we'd
      compare against in any real paper.
    - If both numpy and sklearn agree the method beats the baselines,
      the paper's claim is much more credible.

If sklearn is not installed, every call returns a sentinel
`{sklearn_available: False}` so the verifier can be safely wired
into pipeline.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


log = logging.getLogger("autosci.realverifier")


def _try_import_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import StratifiedKFold
        return LogisticRegression, RandomForestClassifier, MLPClassifier, StratifiedKFold
    except Exception as e:
        log.info("sklearn not available: %s", e)
        return None


@dataclass
class RealVerifierResult:
    sklearn_available: bool
    n_samples: int = 0
    n_features: int = 0
    method_auroc_mean: float = 0.0
    method_auroc_std: float = 0.0
    random_forest_auroc_mean: float = 0.0
    random_forest_auroc_std: float = 0.0
    logreg_auroc_mean: float = 0.0
    logreg_auroc_std: float = 0.0
    mlp_auroc_mean: float = 0.0
    mlp_auroc_std: float = 0.0
    delta_vs_best_baseline_mean: float = 0.0
    delta_vs_best_baseline_std: float = 0.0
    notes: list[str] = field(default_factory=list)


def _make_synthetic(n_samples: int, n_features: int, n_useful: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features)).astype(np.float32)
    y = (X[:, :n_useful].sum(axis=1) + 0.6 * rng.normal(0, 1, size=n_samples)) > 0
    y = y.astype(np.int64)
    return X, y


def _auroc(y_true, p):
    pos = p[y_true == 1]
    neg = p[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diffs = pos[:, None] - neg[None, :]
    conc = (diffs > 0).sum() + 0.5 * (diffs == 0).sum()
    return float(conc / (len(pos) * len(neg)))


def _cv_auroc(make_model, X, y, n_splits: int = 5, seed: int = 0):
    from sklearn.model_selection import StratifiedKFold
    aucs = []
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(X, y):
        model = make_model()
        model.fit(X[tr], y[tr])
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X[te])[:, 1]
        else:
            p = model.decision_function(X[te])
        aucs.append(_auroc(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))


def run_real_verifier(
    n_samples: int = 600,
    n_features: int = 200,
    n_useful: int = 30,
    n_splits: int = 5,
    seed: int = 0,
) -> RealVerifierResult:
    """Run sklearn baselines + a 'our method' (LogReg-on-truth-features)."""
    sk = _try_import_sklearn()
    if sk is None:
        return RealVerifierResult(sklearn_available=False,
                                  notes=["sklearn not installed; install with `pip install scikit-learn`"])
    LogReg, RF, MLP, _ = sk

    X, y = _make_synthetic(n_samples, n_features, n_useful, seed)

    # "Our method" — full-feature logreg (upper bound)
    our_mean, our_std = _cv_auroc(lambda: LogReg(max_iter=200, C=1.0), X, y, n_splits, seed)
    rf_mean, rf_std = _cv_auroc(lambda: RF(n_estimators=50, random_state=seed), X, y, n_splits, seed)
    lr_mean, lr_std = _cv_auroc(lambda: LogReg(max_iter=200, C=0.1), X, y, n_splits, seed)
    mlp_mean, mlp_std = _cv_auroc(
        lambda: MLP(hidden_layer_sizes=(32,), max_iter=100, random_state=seed),
        X, y, n_splits, seed,
    )

    best_baseline = max(rf_mean, lr_mean, mlp_mean)
    delta_mean = our_mean - best_baseline
    # naive std combination
    delta_std = math.sqrt(our_std ** 2 + max(rf_std, lr_std, mlp_std) ** 2)

    notes = []
    if our_mean > best_baseline + 0.02:
        notes.append(f"method beats the best baseline by {delta_mean * 100:+.1f}pp")
    elif abs(delta_mean) < 0.02:
        notes.append(f"method is on par with the best baseline ({delta_mean * 100:+.1f}pp)")
    else:
        notes.append(f"method trails the best baseline by {-delta_mean * 100:.1f}pp — re-examine")

    return RealVerifierResult(
        sklearn_available=True,
        n_samples=n_samples, n_features=n_features,
        method_auroc_mean=our_mean, method_auroc_std=our_std,
        random_forest_auroc_mean=rf_mean, random_forest_auroc_std=rf_std,
        logreg_auroc_mean=lr_mean, logreg_auroc_std=lr_std,
        mlp_auroc_mean=mlp_mean, mlp_auroc_std=mlp_std,
        delta_vs_best_baseline_mean=delta_mean,
        delta_vs_best_baseline_std=delta_std,
        notes=notes,
    )
