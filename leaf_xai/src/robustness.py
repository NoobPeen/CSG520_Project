"""
V2 -- Stability: "does the answer survive a small, harmless change to the photo?"

For every arm, the same stratified sample of TEST images is re-classified
under four kinds of change, each at several strengths:

    brightness  x0.6  x0.8  x1.2  x1.4
    contrast    x0.6  x0.8  x1.2  x1.4
    rotation    -30  -15  +15  +30 degrees (reflect padding, no black corners)
    blur        Gaussian sigma 1, 2, 3 px

The change is applied to the ORIGINAL photograph, then each arm gets its own
evaluation view (deep nets: resize 255 + centre-crop 224; A2: whole frame at
256), so every arm is shown the same altered photograph.

Metrics (per arm, per change type and per strength):
    flip rate          share of images whose predicted label differs from the
                       clean prediction (right or wrong does not matter)
    flip rate | clean-correct   the same, only over images the arm got right
    accuracy retention accuracy under the change / clean accuracy
    confidence drop    mean fall in the probability of the clean-predicted class
Lower flip rate and retention close to 1 are better.

All inference is fp32. Progress is saved after every condition, so a killed
run resumes.

    python src/robustness.py --arms a1_resnet50 a3_cnn a4_vit
    python src/robustness.py --arms a2_rf --n-jobs -1          # CPU runtime
    python src/robustness.py --summarize                       # v2_table.csv + figure

Outputs: <out>/robustness/<arm>/{per_image.csv, levels.csv, summary.json},
<out>/robustness/{sample.csv, v2_table.csv, v2_levels.csv, v2_pairwise.csv,
v2_flip_curves.png}.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import config
import data as data_module
import views

ARMS = ("a1_resnet50", "a1_resnet50_scratch", "a1_efficientnet_b0",
        "a2_rf", "a3_cnn", "a4_vit")
KINDS = tuple(views.PERTURBATIONS)


# ---------------------------------------------------------------------------
# Sample and model
# ---------------------------------------------------------------------------
def select_sample(split_file: Path, per_class: int, seed: int) -> pd.DataFrame:
    """
    Same rule as explain.select_sample: one rng, one permutation per class in
    sorted label order, first `per_class` taken. The permutations do not depend
    on `per_class`, so the 10/class V1 sample is the first 10 of these 20.
    """
    test = data_module.get_split("test", split_file)
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in test.groupby("label", sort=True):
        idx = rng.permutation(len(group))[:per_class]
        parts.append(group.iloc[idx])
    return pd.concat(parts).reset_index(drop=True)[["relpath", "label"]]


def load_predictor(arm: str, args, classes: list[str]):
    import explain                                     # reuse the V1 wrappers
    pred = explain.load_arm(arm, args.ckpt_dir, classes, device=args.device, n_jobs=args.n_jobs)
    if hasattr(pred, "use_amp"):
        pred.use_amp = False                           # fp32, as in the V1 rerun
    return pred


def view_fn(arm: str, img_size: int):
    if arm.startswith("a2_"):
        return views.a2_view
    return lambda img: views.dl_view(img, img_size)


def _atomic_pickle(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Run one arm
# ---------------------------------------------------------------------------
def run_arm(arm: str, args, sample: pd.DataFrame, classes: list[str], out_root: Path) -> dict:
    arm_dir = out_root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    state_path = arm_dir / "state.pkl"
    state = {"relpaths": list(sample["relpath"]), "probs": {}}
    if state_path.exists() and not args.fresh:
        saved = pickle.loads(state_path.read_bytes())
        if saved.get("relpaths") == state["relpaths"]:
            state = saved
            print(f"[{arm}] resuming: {len(state['probs'])} conditions done")
        else:
            print(f"[{arm}] saved state is for a different sample; starting over")

    todo = [c for c in views.conditions() if views.condition_name(*c) not in state["probs"]]
    if todo:
        predict = load_predictor(arm, args, classes)
        root = Path(args.data_root)
        with ThreadPoolExecutor(max_workers=8) as pool:
            originals = list(pool.map(lambda r: views.load_original(root / r), sample["relpath"]))
        to_view = view_fn(arm, args.img_size)

        # Built-in check: the dl view must equal trainer's evaluation geometry.
        if not arm.startswith("a2_"):
            import trainer
            ref = trainer.load_eval_image_uint8(root / sample["relpath"].iloc[0], args.img_size)
            diff = int(np.abs(ref.astype(int) - to_view(originals[0]).astype(int)).max())
            if diff > 1:
                raise RuntimeError(f"dl_view differs from trainer.load_eval_image_uint8 by {diff}")

        for kind, level in todo:
            name = views.condition_name(kind, level)
            started = time.perf_counter()
            batch = np.stack([to_view(views.perturb(img, kind, level)) for img in originals])
            probs = np.asarray(predict.predict_proba(batch), dtype=np.float32)
            state["probs"][name] = probs
            _atomic_pickle(state, state_path)
            acc = float((probs.argmax(1) == _true_idx(sample, classes)).mean())
            print(f"[{arm}] {name:<16} acc={acc:.3f}  ({time.perf_counter() - started:.0f}s)",
                  flush=True)

    per_image = build_per_image(arm, sample, classes, state["probs"])
    per_image.to_csv(arm_dir / "per_image.csv", index=False)
    summary, levels = summarize_arm(arm, per_image, args.seed)
    levels.to_csv(arm_dir / "levels.csv", index=False)
    (arm_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(summary)
    return summary


def _true_idx(sample: pd.DataFrame, classes: list[str]) -> np.ndarray:
    lookup = {c: i for i, c in enumerate(classes)}
    return sample["label"].map(lookup).to_numpy()


def build_per_image(arm, sample, classes, probs_by_cond) -> pd.DataFrame:
    """Long table: one row per (image, condition)."""
    true = _true_idx(sample, classes)
    clean = probs_by_cond["clean"]
    clean_pred = clean.argmax(1)
    rows = np.arange(len(sample))
    frames = []
    for kind, level in views.conditions():
        name = views.condition_name(kind, level)
        p = probs_by_cond[name]
        pred = p.argmax(1)
        frames.append(pd.DataFrame({
            "relpath": sample["relpath"], "label": sample["label"], "true": true,
            "condition": name, "kind": kind, "level": level,
            "pred": pred, "conf": p.max(1), "p_true": p[rows, true],
            "p_clean_pred": p[rows, clean_pred],
            "clean_pred": clean_pred, "clean_correct": clean_pred == true,
            "correct": pred == true, "flip": pred != clean_pred,
            "conf_drop": clean[rows, clean_pred] - p[rows, clean_pred],
        }))
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "arm", arm)
    return out


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def _boot_ci(values: np.ndarray, seed: int, n_boot: int = 2000) -> list[float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(1)
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def summarize_arm(arm: str, df: pd.DataFrame, seed: int) -> tuple[dict, pd.DataFrame]:
    clean = df[df["kind"] == "clean"]
    clean_acc = float(clean["correct"].mean())
    pert = df[df["kind"] != "clean"]

    levels = (pert.groupby(["kind", "level"], sort=False)
              .agg(flip_rate=("flip", "mean"), accuracy=("correct", "mean"),
                   conf_drop=("conf_drop", "mean")).reset_index())
    levels["retention"] = levels["accuracy"] / clean_acc
    levels.insert(0, "arm", arm)

    def block(d: pd.DataFrame) -> dict:
        # CI over images: average each image's flips across the block's levels,
        # then bootstrap those per-image means (images are the independent unit).
        per_img = d.groupby("relpath")["flip"].mean().to_numpy()
        cc = d[d["clean_correct"]]
        return {
            "flip_rate": float(d["flip"].mean()),
            "flip_rate_ci95": _boot_ci(per_img, seed),
            "flip_rate_clean_correct": float(cc["flip"].mean()) if len(cc) else float("nan"),
            "accuracy": float(d["correct"].mean()),
            "retention": float(d["correct"].mean() / clean_acc) if clean_acc else float("nan"),
            "conf_drop": float(d["conf_drop"].mean()),
            "worst_level_flip_rate": float(d.groupby("level")["flip"].mean().max()),
        }

    summary = {
        "arm": arm, "n_images": int(clean["relpath"].nunique()),
        "clean_accuracy": clean_acc,
        "mean_clean_confidence": float(clean["conf"].mean()),
        "kinds": {k: block(pert[pert["kind"] == k]) for k in KINDS if (pert["kind"] == k).any()},
        "overall": block(pert),
        "perturbations": {k: list(v) for k, v in views.PERTURBATIONS.items()},
    }
    return summary, levels


def print_summary(s: dict) -> None:
    print(f"\n=== V2 / {s['arm']} ===  images={s['n_images']}  clean acc={s['clean_accuracy']:.3f}")
    for name, b in list(s["kinds"].items()) + [("OVERALL", s["overall"])]:
        lo, hi = b["flip_rate_ci95"]
        print(f"  {name:<11} flip {b['flip_rate']:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"retention {b['retention']:.3f}  conf drop {b['conf_drop']:+.3f}  "
              f"worst level {b['worst_level_flip_rate']:.3f}")


def summarize_all(out_root: Path) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    summaries = [json.loads(p.read_text()) for p in sorted(out_root.glob("*/summary.json"))]
    if not summaries:
        print("[robustness] no summaries yet")
        return pd.DataFrame()

    rows = []
    for s in summaries:
        row = {"arm": s["arm"], "n_images": s["n_images"], "clean_acc": s["clean_accuracy"]}
        for name, b in list(s["kinds"].items()) + [("overall", s["overall"])]:
            row[f"{name}_flip"] = b["flip_rate"]
            row[f"{name}_flip_ci"] = "[{:.3f}, {:.3f}]".format(*b["flip_rate_ci95"])
            row[f"{name}_retention"] = b["retention"]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_root / "v2_table.csv", index=False)

    levels = pd.concat([pd.read_csv(p) for p in sorted(out_root.glob("*/levels.csv"))])
    levels.to_csv(out_root / "v2_levels.csv", index=False)

    # Paired comparison: same images for every arm, so compare each image's
    # number of flips across all perturbed conditions (Wilcoxon signed-rank).
    flips = {}
    for p in sorted(out_root.glob("*/per_image.csv")):
        d = pd.read_csv(p)
        d = d[d["kind"] != "clean"]
        flips[d["arm"].iloc[0]] = d.groupby("relpath")["flip"].sum()
    pairs = []
    for a, b in itertools.combinations(sorted(flips), 2):
        common = flips[a].index.intersection(flips[b].index)
        x, y = flips[a][common], flips[b][common]
        diff = (x - y).to_numpy()
        p = float(wilcoxon(x, y).pvalue) if np.any(diff != 0) else 1.0
        pairs.append({"arm_a": a, "arm_b": b, "n": len(common),
                      "mean_flips_a": float(x.mean()), "mean_flips_b": float(y.mean()),
                      "diff_a_minus_b": float(diff.mean()), "wilcoxon_p": p})
    pd.DataFrame(pairs).to_csv(out_root / "v2_pairwise.csv", index=False)

    cols = ["arm", "clean_acc"] + [f"{k}_flip" for k in KINDS if f"{k}_flip" in table] + \
           ["overall_flip", "overall_retention"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if pairs:
        print("\npaired (flips per image, all perturbed conditions):")
        print(pd.DataFrame(pairs).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    try:
        plot_curves(levels, out_root / "v2_flip_curves.png")
    except Exception as exc:
        print(f"[robustness] figure skipped: {exc}")
    print(f"\n[robustness] wrote {out_root / 'v2_table.csv'}")
    return table


def plot_curves(levels: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = [k for k in KINDS if (levels["kind"] == k).any()]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.6 * len(kinds), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    neutral = {"brightness": 1.0, "contrast": 1.0, "rotation": 0.0, "blur": 0.0}
    for ax, kind in zip(axes, kinds):
        for arm, d in levels[levels["kind"] == kind].groupby("arm"):
            d = d.sort_values("level")
            # add the clean point (flip rate 0 by definition) at the neutral level
            x = np.r_[d["level"].to_numpy(), neutral[kind]]
            y = np.r_[d["flip_rate"].to_numpy(), 0.0]
            order = np.argsort(x)
            ax.plot(x[order], y[order], marker="o", ms=3, label=arm)
        ax.set_title(kind)
        ax.set_xlabel({"rotation": "degrees", "blur": "sigma (px)"}.get(kind, "factor"))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("label flip rate")
    axes[-1].legend(fontsize=7, frameon=False)
    fig.suptitle("V2: share of predictions that change under each perturbation", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[robustness] saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="V2: prediction stability under small image changes")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=[])
    p.add_argument("--summarize", action="store_true",
                   help="only (re)build v2_table.csv and the figure from saved results")
    p.add_argument("--per-class", type=int, default=20, help="20 x 38 = 760 test images")
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=config.RAW_DIR)
    p.add_argument("--splits", type=Path, default=config.SPLIT_FILE)
    p.add_argument("--classes", type=Path, default=None)
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--n-jobs", type=int, default=-1, help="CPU workers for A2 features")
    p.add_argument("--limit", type=int, default=0, help="first N images only (smoke test)")
    p.add_argument("--fresh", action="store_true", help="ignore saved progress for these arms")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is not None:
        import trainer
        trainer.set_output_root(args.output_dir)
    out_root = Path(config.OUTPUT_DIR) / "robustness"
    out_root.mkdir(parents=True, exist_ok=True)
    if args.ckpt_dir is None:
        args.ckpt_dir = config.CKPT_DIR

    if args.arms:
        import trainer
        classes = trainer.resolve_classes(args.splits, args.classes)
        sample = select_sample(args.splits, args.per_class, args.seed)
        if args.limit:
            sample = sample.head(args.limit)
        sample.to_csv(out_root / "sample.csv", index=False)
        print(f"[robustness] {len(sample)} test images ({args.per_class}/class), "
              f"{len(views.conditions()) - 1} perturbed conditions + clean")
        for arm in args.arms:
            run_arm(arm, args, sample, classes, out_root)

    summarize_all(out_root)


if __name__ == "__main__":
    main()
