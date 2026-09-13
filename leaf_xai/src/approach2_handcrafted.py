"""
APPROACH 2 -- Hand-picked features + SVM and Random Forest.

The contrast with Approach 1 is the point. There, the representation is
learned and we have to ask a saliency method where the network looked. Here we
choose the representation ourselves -- 114 colour numbers, 58 texture numbers,
14 shape numbers -- so the model's evidence is legible by construction: a
feature importance is a statement about a quantity we defined.

Pipeline:
  1. features.extract_dataset() turns each image into 186 numbers (cached).
  2. Model selection uses the SAME validation split as Approach 1, wired into
     scikit-learn through PredefinedSplit. No k-fold reshuffling, so the two
     approaches are tuned and tested on identical images.
  3. Both models are refit on train+val and scored once on the test split.
  4. Importances are reported per feature (Random Forest) and per feature
     group by permutation (both models), which is the number that answers
     "is it the colour, the texture or the shape doing the work?".

Typical run:

    python src/data.py --build
    python src/approach2_handcrafted.py --max-per-class 400
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
import data as data_module
import features as feature_module
import utils

FEATURE_GROUPS = ("color", "texture", "shape")


# --------------------------------------------------------------------------
# Feature matrices for the three splits
# --------------------------------------------------------------------------
def load_feature_matrices(args) -> dict:
    """
    Build (or read from cache) X and y for train/val/test.

    Labels are mapped to indices through data/classes.json, the same mapping
    Approach 1 uses, so a confusion matrix from either approach can be read
    with the same axis labels.
    """
    classes = config.load_classes()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    out = {"classes": classes}

    for split in ("train", "val", "test"):
        df = data_module.get_split(split)
        if args.max_per_class:
            df = data_module.subsample_per_class(df, args.max_per_class)

        tag = f"a2_{split}_n{args.max_per_class or 'all'}_s{args.img_size}"
        cache_path = config.FEATURE_DIR / f"{tag}.npz"

        X, y_labels, names, relpaths = feature_module.extract_dataset(
            df,
            root=config.RAW_DIR,
            n_jobs=args.n_jobs,
            img_size=args.img_size,
            cache_path=cache_path,
            force=args.force_features,
        )
        y = np.array([class_to_idx[label] for label in y_labels], dtype=np.int64)
        out[split] = {"X": X, "y": y, "relpaths": relpaths}
        out["feature_names"] = names

    return out


def make_predefined_split(n_train: int, n_val: int) -> PredefinedSplit:
    """
    One fold: train rows get -1 (never validated on), val rows get 0.
    GridSearchCV then scores each candidate on our real validation split and
    refits the winner on train+val.
    """
    test_fold = np.concatenate([np.full(n_train, -1), np.zeros(n_val)])
    return PredefinedSplit(test_fold=test_fold)


# --------------------------------------------------------------------------
# Model definitions
# --------------------------------------------------------------------------
def svm_search_space(quick: bool) -> tuple[Pipeline, dict]:
    """
    RBF-kernel SVM. Standardisation is mandatory here: histogram bins live in
    [0, 1] while colour moments run to 255, and an RBF kernel measures plain
    Euclidean distance, so without scaling the moments would drown everything
    else. class_weight='balanced' compensates for the class imbalance.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced", cache_size=1000)),
    ])
    grid = ({"svm__C": [10.0], "svm__gamma": ["scale"]} if quick else
            {"svm__C": [1.0, 10.0, 100.0], "svm__gamma": ["scale", 0.01, 0.001]})
    return pipeline, grid


def rf_search_space(quick: bool, seed: int) -> tuple[Pipeline, dict]:
    """
    Random Forest. No scaling needed -- trees split on thresholds, so feature
    magnitude is irrelevant. Kept in a Pipeline anyway so both models are
    handled by identical code downstream.
    """
    pipeline = Pipeline([
        ("rf", RandomForestClassifier(
            n_estimators=300 if quick else 500,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )),
    ])
    grid = ({"rf__max_depth": [None]} if quick else
            {"rf__max_depth": [None, 25], "rf__max_features": ["sqrt", 0.2],
             "rf__min_samples_leaf": [1, 2]})
    return pipeline, grid


def fit_with_search(name, pipeline, grid, matrices, seed) -> tuple[Pipeline, dict]:
    """Grid search on the fixed validation split, then refit on train+val."""
    X_train, y_train = matrices["train"]["X"], matrices["train"]["y"]
    X_val, y_val = matrices["val"]["X"], matrices["val"]["y"]

    X_search = np.vstack([X_train, X_val])
    y_search = np.concatenate([y_train, y_val])
    cv = make_predefined_split(len(y_train), len(y_val))

    search = GridSearchCV(
        pipeline, grid, scoring="f1_macro", cv=cv,
        refit=True, n_jobs=1, verbose=1,
    )

    print(f"\n[a2] {name}: grid search over "
          f"{np.prod([len(v) for v in grid.values()])} configuration(s) ...")
    started = time.perf_counter()
    search.fit(X_search, y_search)
    elapsed = time.perf_counter() - started

    info = {
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "best_val_macro_f1": float(search.best_score_),
        "search_seconds": round(elapsed, 1),
    }
    print(f"[a2] {name}: best val macro-F1 ={search.best_score_:.4f} "
          f"with {search.best_params_}  ({elapsed:.0f}s)")
    return search.best_estimator_, info


# --------------------------------------------------------------------------
# Interpretability
# --------------------------------------------------------------------------
def group_permutation_importance(model, X, y, feature_names, n_repeats=5,
                                 seed=config.SEED) -> pd.DataFrame:
    """
    Permute an entire feature family at once and measure the macro-F1 drop.

    Permuting whole groups rather than single columns is the honest way to
    read these features: the 32 hue-histogram bins are strongly correlated, so
    shuffling one at a time lets the others carry the signal and every
    individual importance looks negligible. The rows of a group are permuted
    jointly, which preserves the correlations inside the group and destroys
    only its relationship with the label.
    """
    rng = np.random.default_rng(seed)
    baseline = f1_score(y, model.predict(X), average="macro", zero_division=0)

    rows = []
    for group in FEATURE_GROUPS:
        columns = [i for i, n in enumerate(feature_names)
                   if feature_module.feature_group(n) == group]
        if not columns:
            continue
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            order = rng.permutation(len(X_perm))
            X_perm[:, columns] = X_perm[order][:, columns]
            score = f1_score(y, model.predict(X_perm), average="macro", zero_division=0)
            drops.append(baseline - score)
        rows.append({
            "group": group,
            "n_features": len(columns),
            "mean_f1_drop": float(np.mean(drops)),
            "std_f1_drop": float(np.std(drops)),
        })

    table = pd.DataFrame(rows).sort_values("mean_f1_drop", ascending=False)
    table.attrs["baseline_macro_f1"] = float(baseline)
    return table


def report_importances(model_name, model, matrices, feature_names, args) -> dict:
    """Per-feature importance (forests only) plus group permutation importance."""
    result: dict = {}
    X_test, y_test = matrices["test"]["X"], matrices["test"]["y"]

    # --- impurity-based importance, available for tree ensembles only ------
    estimator = model.named_steps.get("rf")
    if estimator is not None:
        importances = estimator.feature_importances_
        top = np.argsort(-importances)[:25]
        utils.plot_horizontal_bars(
            labels=[feature_names[i] for i in top],
            values=importances[top],
            title="Random Forest: 25 most important features (impurity)",
            out_name="a2_rf_feature_importance",
            xlabel="mean decrease in impurity",
        )
        result["top_features"] = [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in top
        ]
        # Aggregate to families so the report can say what kind of evidence
        # the forest actually relies on.
        per_group = {
            g: float(sum(importances[i] for i, n in enumerate(feature_names)
                         if feature_module.feature_group(n) == g))
            for g in FEATURE_GROUPS
        }
        result["impurity_importance_by_group"] = per_group
        print(f"[a2] {model_name} impurity importance by group: "
              + ", ".join(f"{g}={v:.3f}" for g, v in per_group.items()))

    # --- permutation importance by group ----------------------------------
    print(f"[a2] {model_name}: group permutation importance "
          f"({args.perm_repeats} repeats on the test split) ...")
    table = group_permutation_importance(
        model, X_test, y_test, feature_names,
        n_repeats=args.perm_repeats, seed=args.seed,
    )
    print(table.to_string(index=False))

    utils.plot_horizontal_bars(
        labels=table["group"].tolist(),
        values=table["mean_f1_drop"].to_numpy(),
        errors=table["std_f1_drop"].to_numpy(),
        title=f"{model_name}: macro-F1 drop when a feature family is permuted",
        out_name=f"a2_{model_name.lower()}_group_permutation",
        xlabel="drop in macro F1",
    )
    result["group_permutation_importance"] = table.to_dict(orient="records")
    result["permutation_baseline_macro_f1"] = table.attrs["baseline_macro_f1"]
    table.to_csv(config.METRIC_DIR / f"a2_{model_name.lower()}_group_permutation.csv",
                 index=False)
    return result


# --------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict:
    utils.set_seed(args.seed)
    config.ensure_dirs()

    matrices = load_feature_matrices(args)
    classes = matrices["classes"]
    feature_names = matrices["feature_names"]
    print(f"[a2] {len(classes)} classes | "
          f"train={len(matrices['train']['y'])} val={len(matrices['val']['y'])} "
          f"test={len(matrices['test']['y'])} | {len(feature_names)} features")

    if args.segmentation_figure:
        feature_module.save_segmentation_examples(data_module.get_split("test"))

    all_metrics = {}
    models_to_run = []
    if not args.skip_svm:
        models_to_run.append(("SVM", *svm_search_space(args.quick)))
    if not args.skip_rf:
        models_to_run.append(("RF", *rf_search_space(args.quick, args.seed)))

    for name, pipeline, grid in models_to_run:
        model, search_info = fit_with_search(name, pipeline, grid, matrices, args.seed)

        X_test, y_test = matrices["test"]["X"], matrices["test"]["y"]
        y_pred = model.predict(X_test)
        metrics = utils.compute_metrics(y_test, y_pred, classes)
        metrics.update({
            "approach": "A2_handcrafted_features",
            "model": name,
            "n_features": len(feature_names),
            "max_per_class": args.max_per_class,
            "feature_img_size": args.img_size,
            **search_info,
        })

        utils.print_summary(metrics, f"A2 / {name} / test set")
        utils.plot_confusion_matrix(y_test, y_pred, classes,
                                    f"a2_{name.lower()}_confusion")
        metrics["interpretability"] = report_importances(
            name, model, matrices, feature_names, args
        )

        utils.save_metrics(metrics, f"a2_{name.lower()}")
        model_path = config.CKPT_DIR / f"a2_{name.lower()}.joblib"
        dump(model, model_path)
        print(f"[a2] saved model -> {model_path}")
        all_metrics[name] = metrics

    # Side-by-side line for the results table in the report.
    if all_metrics:
        print("\n=== A2 summary ===")
        for name, m in all_metrics.items():
            print(f"  {name:>3}: accuracy={m['accuracy']:.4f}  "
                  f"macro-F1={m['macro_f1']:.4f}  "
                  f"balanced-acc={m['balanced_accuracy']:.4f}")

    return all_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approach 2: hand-crafted features with SVM and Random Forest"
    )
    parser.add_argument("--max-per-class", type=int, default=config.A2_MAX_PER_CLASS,
                        help="cap images per class (0 = use every image; slow)")
    parser.add_argument("--img-size", type=int, default=config.A2_FEATURE_IMG_SIZE)
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="parallel workers for feature extraction")
    parser.add_argument("--perm-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--quick", action="store_true",
                        help="single-point hyperparameter grids, for a fast pass")
    parser.add_argument("--force-features", action="store_true",
                        help="ignore the cached .npz files and re-extract")
    parser.add_argument("--skip-svm", action="store_true")
    parser.add_argument("--skip-rf", action="store_true")
    parser.add_argument("--segmentation-figure", action="store_true",
                        help="save a figure showing the leaf masks")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
