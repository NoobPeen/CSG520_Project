"""
V3 -- Confidence: "is the model sure when it is right and unsure when it is wrong?"

Computed from each arm's full test-set probability matrix; no image is
perturbed and nothing is retrained.

Where the probabilities come from:
  * A3, A4                preds/<arm>_test_probs.npz (written by trainer.final_report)
  * A1 arms               A1 predates that export, so this script scores the saved
                          checkpoint on the TEST split once (fp32, evaluation geometry)
                          and writes preds/<arm>_test_probs.npz in the same format.
  * A2 RF                 the cached test features (features/a2_test_n400_s256.npz, the
                          exact rows behind A2's reported metrics) + a2_rf.joblib.
                          The SVM was fitted without probability=True, so it has no
                          V3 row (consistent with "RF carries the verticals").
Each source prints its macro-F1 next to the value in metrics/<arm>.json, so a
mismatch (wrong checkpoint, wrong split) is visible immediately.

Metrics per arm:
  mean confidence when correct / when wrong, and the gap (bootstrap 95% CI)
  confident mistakes: share (and count, Wilson CI) of errors with confidence > 0.9
  AUROC of confidence for separating right from wrong (threshold-free)
  ECE (15 equal-width bins), NLL, Brier score
Reported on each arm's full test set and on the "common" subset: the 6,953
test images A2 was scored on, which every other arm also covers.

Reading the numbers (put these next to the table):
  * Label smoothing 0.05 over 38 classes caps a well-trained deep net's
    confidence near 1 - 0.05 + 0.05/38 = 0.951. So "> 0.9" is close to the
    most confident the deep nets ever get, and ECE for them partly measures
    the smoothing, not only miscalibration.
  * RF confidence is a vote share across 500 trees, which rarely reaches 0.9.
    Few confident mistakes for A2 is partly a property of how RF confidence is
    formed. AUROC is the fairer cross-family comparison.
  * The deep nets make only ~20 errors on 8,146 images, so their "wrong" means
    rest on very few images: always print n next to them.

    python src/confidence.py --arms a1_resnet50 a1_efficientnet_b0 a1_resnet50_scratch a3_cnn a4_vit a2_rf
    python src/confidence.py --summarize
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

import config
import data as data_module

ARMS = ("a1_resnet50", "a1_resnet50_scratch", "a1_efficientnet_b0",
        "a2_rf", "a3_cnn", "a4_vit")
HIGH = 0.9
N_BINS = 15


# ---------------------------------------------------------------------------
# Probability sources
# ---------------------------------------------------------------------------
def preds_path(arm: str) -> Path:
    return Path(config.OUTPUT_DIR) / "preds" / f"{arm}_test_probs.npz"


def load_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {"probs": z["probs"].astype(np.float32), "y_true": z["y_true"].astype(int),
            "relpaths": [str(r) for r in z["relpaths"]], "classes": [str(c) for c in z["classes"]]}


def save_npz(path: Path, probs, y_true, relpaths, classes, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probs=probs.astype(np.float32), y_true=np.asarray(y_true),
                        y_pred=probs.argmax(1), relpaths=np.array(relpaths),
                        classes=np.array(classes), source=np.array(source))
    print(f"[confidence] wrote {path}")


def export_dl(arm: str, args, classes: list[str]) -> dict:
    """Score a deep-net checkpoint on the full TEST split (fp32) and save the npz."""
    import explain
    import trainer
    pred = explain.load_arm(arm, args.ckpt_dir, classes, device=args.device)
    if hasattr(pred, "use_amp"):
        pred.use_amp = False
    test = data_module.get_split("test", args.splits)
    if args.limit:
        test = test.head(args.limit)
    lookup = {c: i for i, c in enumerate(classes)}
    y_true = test["label"].map(lookup).to_numpy()
    root = Path(args.data_root)
    rels = list(test["relpath"])
    chunks = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(rels), 256):
            imgs = list(pool.map(lambda r: trainer.load_eval_image_uint8(root / r, args.img_size),
                                 rels[start:start + 256]))
            chunks.append(np.asarray(pred.predict_proba(np.stack(imgs)), dtype=np.float32))
            print(f"[{arm}] scored {min(start + 256, len(rels))}/{len(rels)}", end="\r", flush=True)
    print()
    probs = np.concatenate(chunks)
    save_npz(preds_path(arm), probs, y_true, rels, classes, "confidence.py fp32 export")
    return {"probs": probs, "y_true": y_true, "relpaths": rels, "classes": classes}


def export_a2(arm: str, args, classes: list[str]) -> dict:
    """A2 RF on its cached test features: the exact rows behind its reported metrics."""
    from joblib import load
    feats = np.load(args.a2_features, allow_pickle=True)
    X = np.nan_to_num(feats["X"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lookup = {c: i for i, c in enumerate(classes)}
    y_true = np.array([lookup[str(l)] for l in feats["y"]])
    model = load(Path(args.ckpt_dir) / f"{arm}.joblib")
    if not hasattr(model, "predict_proba"):
        raise RuntimeError(f"{arm} has no predict_proba (SVC without probability=True)")
    est = getattr(model, "steps", [[None, model]])[-1][1]
    if hasattr(est, "n_jobs"):
        est.n_jobs = args.n_jobs
    raw = model.predict_proba(X)
    probs = np.zeros((len(X), len(classes)), dtype=np.float32)
    probs[:, np.asarray(model.classes_).astype(int)] = raw      # sklearn column order
    rels = [str(r) for r in feats["relpaths"]]
    save_npz(preds_path(arm), probs, y_true, rels, classes, f"cached features {Path(args.a2_features).name}")
    return {"probs": probs, "y_true": y_true, "relpaths": rels, "classes": classes}


def get_probs(arm: str, args, classes: list[str]) -> dict | None:
    path = preds_path(arm)
    if path.exists() and not args.re_export:
        d = load_npz(path)
        if d["classes"] != list(classes):
            raise ValueError(f"{path.name}: class order differs from classes.json")
        return d
    if args.no_export:
        print(f"[confidence] {arm}: no {path.name}, skipping (--no-export)")
        return None
    return export_a2(arm, args, classes) if arm.startswith("a2_") else export_dl(arm, args, classes)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> tuple[float, float, pd.DataFrame]:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    rows, total, worst = [], 0.0, 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        gap = abs(correct[m].mean() - conf[m].mean())
        total += m.mean() * gap
        worst = max(worst, gap)
        rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "n": int(m.sum()),
                     "mean_conf": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
    return float(total), float(worst), pd.DataFrame(rows)


def confidence_metrics(probs: np.ndarray, y_true: np.ndarray, seed: int, n_boot: int = 2000) -> dict:
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = pred == y_true
    wrong = ~correct
    n, n_wrong = len(y_true), int(wrong.sum())
    k_high = int((conf[wrong] > HIGH).sum())

    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        c, w = correct[i], wrong[i]
        if c.any() and w.any():
            gaps.append(conf[i][c].mean() - conf[i][w].mean())
    ece_v, mce_v, _ = ece(conf, correct.astype(float))
    p_true = np.clip(probs[np.arange(n), y_true], 1e-12, 1.0)
    onehot = np.eye(probs.shape[1])[y_true]

    return {
        "n": n, "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "n_wrong": n_wrong,
        "conf_correct_mean": float(conf[correct].mean()) if correct.any() else float("nan"),
        "conf_correct_median": float(np.median(conf[correct])) if correct.any() else float("nan"),
        "conf_wrong_mean": float(conf[wrong].mean()) if n_wrong else float("nan"),
        "conf_wrong_median": float(np.median(conf[wrong])) if n_wrong else float("nan"),
        "gap": float(conf[correct].mean() - conf[wrong].mean()) if n_wrong and correct.any() else float("nan"),
        "gap_ci95": ([float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]
                     if gaps else [float("nan"), float("nan")]),
        "wrong_above_0.9_n": k_high,
        "wrong_above_0.9_share": k_high / n_wrong if n_wrong else float("nan"),
        "wrong_above_0.9_ci95": wilson(k_high, n_wrong),
        "correct_above_0.9_share": float((conf[correct] > HIGH).mean()) if correct.any() else float("nan"),
        "auroc_right_vs_wrong": (float(roc_auc_score(correct, conf))
                                 if 0 < n_wrong < n else float("nan")),
        "ece": ece_v, "mce": mce_v,
        "nll": float(-np.log(p_true).mean()),
        "brier": float(((probs - onehot) ** 2).sum(1).mean()),
        "max_conf_seen": float(conf.max()),
    }


def metrics_json_f1(arm: str) -> float | None:
    path = Path(config.OUTPUT_DIR) / "metrics" / f"{arm}.json"
    if path.exists():
        m = json.loads(path.read_text())
        return m.get("macro_f1")
    return None


# ---------------------------------------------------------------------------
# Per arm and table
# ---------------------------------------------------------------------------
def run_arm(arm: str, args, classes: list[str], out_root: Path, common: set[str] | None) -> list[dict]:
    d = get_probs(arm, args, classes)
    if d is None:
        return []
    probs, y_true, rels = d["probs"], d["y_true"], np.array(d["relpaths"])
    rows = []
    subsets = [("full", np.ones(len(rels), bool))]
    if common is not None and not arm.startswith("a2_"):
        subsets.append(("common", np.isin(rels, list(common))))
    elif arm.startswith("a2_"):
        subsets.append(("common", np.ones(len(rels), bool)))    # A2's test set IS the common set
    for name, m in subsets:
        if not m.any():
            continue
        met = confidence_metrics(probs[m], y_true[m], args.seed)
        rows.append({"arm": arm, "subset": name, **met})

    full = rows[0]
    ref = metrics_json_f1(arm)
    ref_s = f"{ref:.4f}" if ref is not None else "n/a"
    print(f"\n=== V3 / {arm} ===  n={full['n']}  macro-F1={full['macro_f1']:.4f} "
          f"(metrics/{arm}.json: {ref_s})")
    print(f"  confidence  correct {full['conf_correct_mean']:.3f}  wrong {full['conf_wrong_mean']:.3f}  "
          f"gap {full['gap']:.3f} {np.round(full['gap_ci95'], 3).tolist()}  (n wrong = {full['n_wrong']})")
    print(f"  mistakes > 0.9: {full['wrong_above_0.9_n']}/{full['n_wrong']}  "
          f"AUROC {full['auroc_right_vs_wrong']:.3f}  ECE {full['ece']:.3f}  NLL {full['nll']:.3f}")

    conf = probs.max(1)
    pred = probs.argmax(1)
    bad = (pred != y_true) & (conf > HIGH)
    pd.DataFrame({"relpath": rels[bad], "true": [classes[i] for i in y_true[bad]],
                  "pred": [classes[i] for i in pred[bad]], "confidence": conf[bad]}) \
        .sort_values("confidence", ascending=False) \
        .to_csv(out_root / f"{arm}_confident_mistakes.csv", index=False)
    np.savez_compressed(out_root / f"{arm}_conf.npz", conf=conf, correct=pred == y_true)
    return rows


def plot_all(out_root: Path, arms: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have = [a for a in arms if (out_root / f"{a}_conf.npz").exists()]
    if not have:
        return
    fig, axes = plt.subplots(2, len(have), figsize=(3.0 * len(have), 6.0), squeeze=False)
    for j, arm in enumerate(have):
        z = np.load(out_root / f"{arm}_conf.npz")
        conf, correct = z["conf"], z["correct"].astype(float)
        _, _, bins = ece(conf, correct)
        ax = axes[0, j]
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey")
        centres = (bins["bin_lo"] + bins["bin_hi"]) / 2
        ax.bar(centres, bins["accuracy"], width=1 / N_BINS * 0.9, alpha=0.7)
        ax.plot(bins["mean_conf"], bins["accuracy"], "k.", ms=3)   # where the bin's mass sits
        ax.set_title(arm, fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("confidence", fontsize=7)
        if j == 0:
            ax.set_ylabel("accuracy in bin", fontsize=7)
        ax = axes[1, j]
        b = np.linspace(0, 1, 26)
        ax.hist(conf[correct == 1], bins=b, alpha=0.6, label="correct", density=True)
        if (correct == 0).any():
            ax.hist(conf[correct == 0], bins=b, alpha=0.6, label=f"wrong (n={int((correct == 0).sum())})",
                    density=True)
        ax.set_yscale("log")
        ax.axvline(HIGH, color="k", lw=0.6, ls=":")
        ax.set_xlabel("confidence", fontsize=7)
        ax.legend(fontsize=6, frameon=False)
    fig.suptitle("V3: reliability (top) and confidence when right vs wrong (bottom)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_root / "v3_confidence.png", dpi=150)
    plt.close(fig)
    print(f"[confidence] saved {out_root / 'v3_confidence.png'}")


def write_table(rows: list[dict], out_root: Path) -> pd.DataFrame:
    path = out_root / "v3_table.csv"
    table = pd.DataFrame(rows)
    if path.exists():                                  # keep arms scored in other runtimes
        old = pd.read_csv(path)
        old = old[~old["arm"].isin(table["arm"])] if not table.empty else old
        table = pd.concat([old, table], ignore_index=True)
    if table.empty:
        return table
    for col in ("gap_ci95", "wrong_above_0.9_ci95"):
        if col in table:
            table[col] = table[col].apply(lambda v: "[{:.3f}, {:.3f}]".format(*v) if isinstance(v, list) else v)
    table = table.sort_values(["subset", "arm"]).reset_index(drop=True)
    table.to_csv(path, index=False)
    cols = ["arm", "subset", "n", "accuracy", "n_wrong", "conf_correct_mean", "conf_wrong_mean",
            "gap", "wrong_above_0.9_n", "wrong_above_0.9_share", "auroc_right_vs_wrong", "ece"]
    print("\n" + table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n[confidence] wrote {path}")
    return table


def parse_args():
    p = argparse.ArgumentParser(description="V3: confidence when right vs wrong")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=[])
    p.add_argument("--summarize", action="store_true", help="figure + table from saved results only")
    p.add_argument("--no-export", action="store_true", help="use existing preds/*.npz only")
    p.add_argument("--re-export", action="store_true", help="recompute probabilities even if the npz exists")
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=config.RAW_DIR)
    p.add_argument("--splits", type=Path, default=config.SPLIT_FILE)
    p.add_argument("--classes", type=Path, default=None)
    p.add_argument("--a2-features", type=Path, default=None,
                   help="cached A2 test features (default <out>/features/a2_test_n400_s256.npz)")
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--limit", type=int, default=0, help="first N test images (smoke test of the export)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is not None:
        import trainer
        trainer.set_output_root(args.output_dir)
    out_root = Path(config.OUTPUT_DIR) / "confidence"
    out_root.mkdir(parents=True, exist_ok=True)
    if args.ckpt_dir is None:
        args.ckpt_dir = config.CKPT_DIR
    if args.a2_features is None:
        args.a2_features = Path(config.OUTPUT_DIR) / "features" / "a2_test_n400_s256.npz"

    rows: list[dict] = []
    if args.arms:
        import trainer
        classes = trainer.resolve_classes(args.splits, args.classes)
        common = None
        if Path(args.a2_features).exists():
            common = {str(r) for r in np.load(args.a2_features, allow_pickle=True)["relpaths"]}
        for arm in args.arms:
            rows += run_arm(arm, args, classes, out_root, common)
    table = write_table(rows, out_root)
    if not table.empty:
        plot_all(out_root, sorted(table["arm"].unique()))


if __name__ == "__main__":
    main()
