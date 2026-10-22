"""Maintained upstream statistics and date-grouped research stress tests.

arch supplies SPA/RealityCheck; skfolio supplies combinatorial partitions and path
assignment. Mosaic applies interval purging in trading-session positions, never
by individual stock rows. CPCV is a research stress test, not forward performance.
"""
import importlib.metadata
import numpy as np
import pandas as pd


def multiple_comparison(returns, block=4, reps=1000, seed=7):
    from arch.bootstrap import SPA, RealityCheck
    if not returns.columns.is_unique or not returns.index.is_unique or not returns.index.is_monotonic_increasing:
        raise ValueError("multiple-comparison inputs must have unique strategies and ordered unique periods")
    original = len(returns)
    matrix = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").dropna(axis=0, how="any")
    matrix = matrix.loc[:, matrix.std() > 0]
    if len(matrix) < 20 or matrix.shape[1] < 2:
        return None
    if not 1 <= block <= len(matrix) or reps < 1:
        raise ValueError("invalid bootstrap block/repetition count")
    # arch consumes losses: zero benchmark minus negative strategy returns.
    kwargs = dict(block_size=block, reps=reps, bootstrap="stationary", seed=seed)
    spa = SPA(np.zeros(len(matrix)), -matrix, studentize=True, **kwargs)
    rc = RealityCheck(np.zeros(len(matrix)), -matrix, studentize=False, **kwargs)
    spa.compute()
    rc.compute()
    return {"implementation": "arch.bootstrap", "version": importlib.metadata.version("arch"),
            "strategies": matrix.shape[1], "periods": len(matrix), "excluded_periods": original - len(matrix),
            "missing_policy": "complete_common_periods_no_zero_imputation", "block_size": block, "reps": reps, "seed": seed,
            "reality_check_p": float(rc.pvalues["upper"]), "spa_p": float(spa.pvalues["consistent"]),
            "best_strategy": str(matrix.mean().idxmax()), "best_mean": float(matrix.mean().max())}


def purged_combinations(positions, horizon, holdout_position, embargo=5, n_folds=6, n_test_folds=2):
    from skfolio.model_selection import CombinatorialPurgedCV
    pos = np.asarray(positions)
    if pos.ndim != 1 or not np.isfinite(pos).all() or (pos != np.floor(pos)).any() or (np.diff(pos) < 0).any():
        raise ValueError("ordered integer trading-session positions required")
    if horizon < 1 or embargo < 0:
        raise ValueError("positive horizon and nonnegative embargo required")
    eligible = pos + 1 + horizon + embargo < holdout_position
    unique = np.unique(pos[eligible])
    if len(unique) < n_folds * 2:
        return
    cv = CombinatorialPurgedCV(n_folds=n_folds, n_test_folds=n_test_folds)
    path_ids = cv.get_path_ids()
    for split, (train_dates, test_groups) in enumerate(cv.split(unique[:, None])):
        train = eligible & np.isin(pos, unique[train_dates])
        tests = []
        for group in test_groups:
            test_positions = unique[group]
            # Each test group is one contiguous date fold. Purge any intersecting
            # label interval, including touching endpoints; embargo follows its end.
            lo, hi = test_positions.min() + 1, test_positions.max() + 1 + horizon
            overlap = (pos + 1 <= hi) & (pos + 1 + horizon >= lo)
            after = (pos + 1 > hi) & (pos + 1 <= hi + embargo)
            train &= ~(overlap | after)
            tests.append(np.flatnonzero(eligible & np.isin(pos, test_positions)))
        yield np.flatnonzero(train), tests, path_ids[split]


def cpcv_ridge(X, y, dates, positions, horizon, holdout_position, *, embargo=5, n_folds=6, n_test_folds=2, min_train=200):
    from research import fit_predict, ic_series
    y = np.asarray(y)
    paths, splits = {}, []
    for split, (train, tests, path_ids) in enumerate(purged_combinations(positions, horizon, holdout_position, embargo, n_folds, n_test_folds)):
        train = train[np.isfinite(y[train])]
        if len(train) < min_train:
            continue
        test = np.concatenate(tests)
        _, prediction = fit_predict("ridge", X.iloc[train], y[train], X.iloc[test])
        offset = 0
        for group, path in zip(tests, path_ids):
            paths.setdefault(int(path), np.full(len(y), np.nan))[group] = prediction[offset:offset + len(group)]
            offset += len(group)
        ic = ic_series(prediction, y[test], np.asarray(dates)[test])
        splits.append({"split": split, "train_rows": len(train), "test_rows": len(test),
                       "mean_ic": float(ic.mean()) if ic.notna().any() else None})
    path_results = []
    for path, prediction in sorted(paths.items()):
        ic = ic_series(prediction, y, dates)
        path_results.append({"path": path, "scored_dates": int(ic.notna().sum()),
                             "mean_ic": float(ic.mean()) if ic.notna().any() else None})
    return {"status": "COMPUTED" if splits else "NOT_COMPUTED", "implementation": "skfolio.model_selection.CombinatorialPurgedCV",
            "version": importlib.metadata.version("skfolio"), "model": "fixed_ridge", "n_folds": n_folds, "n_test_folds": n_test_folds,
            "horizon_sessions": horizon, "embargo_sessions": embargo, "splits": splits, "paths": path_results,
            "holdout_used": False, "use": "discovery stability diagnostic; future folds may train earlier test folds; not a deployment gate or forward record"}
