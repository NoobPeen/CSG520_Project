# Leaf disease classification — Approaches 1 and 2

Code for A1 (transfer learning) and A2 (hand-picked features + SVM / Random
Forest). Both approaches read the same `data/splits.csv`, so their test numbers
are computed on identical images and can go straight into one results table.

Every path in the code is relative to this folder — unzip it anywhere and run.

## 1. Setup

```bash
pip install -r requirements.txt          # on Colab/Kaggle torch is preinstalled
```

Place the PlantVillage colour images so that each class is its own folder:

```
data/raw/Tomato___Early_blight/0a1b....jpg
data/raw/Tomato___healthy/....jpg
data/raw/Apple___Black_rot/....jpg
...
```

On Kaggle, `plantvillage-dataset/versions/1/color/` already has this layout —
symlink or copy it to `data/raw`.

## 2. Build the split (run this once, before anything else)

```bash
python src/data.py --build --summary
```

Writes `data/splits.csv` (stratified 70/15/15) and `data/classes.json`.
Do not delete these between runs: they are what keeps A1 and A2 comparable.
`--summary` also saves the per-class counts table, which is EDA material for
the report.

## 3. Approach 1 — transfer learning

```bash
python src/approach1_transfer.py --arch resnet50
python src/approach1_transfer.py --arch efficientnet_b0
```

Two-stage schedule: 3 warm-up epochs with the backbone frozen (a random head
would otherwise wreck the pretrained filters), then 12 fine-tuning epochs at
1e-4 with cosine decay and early stopping on validation macro-F1.

Useful flags:

| flag | what it does |
|---|---|
| `--no-pretrained` | random init instead of ImageNet — **this is the ablation that answers A1's question**: run it and compare against the pretrained run |
| `--max-per-class 300` | quick pass on a subset |
| `--eval-only` | re-score the saved checkpoint without training |
| `--no-gradcam` | skip the heatmap figure |
| `--batch-size` `--img-size` `--epochs` `--lr` | the usual |

Rough T4 timing at 224px, full PlantVillage: ~6–8 min/epoch for ResNet50,
~4–5 min for EfficientNet-B0. Budget about 1.5 hours per backbone for the
full 15 epochs, less if early stopping fires.

## 4. Approach 2 — hand-crafted features + SVM / RF

```bash
python src/approach2_handcrafted.py --max-per-class 400 --segmentation-figure
```

186 features per image (114 colour, 58 texture, 14 shape), extracted in
parallel and cached to `outputs/features/*.npz` — the second run skips
extraction entirely. Hyperparameters are selected on the same validation split
A1 uses (wired in through `PredefinedSplit`), then both models are refit on
train+val and scored once on test.

Useful flags: `--quick` (single-point grids), `--max-per-class 0` (use every
image, slow), `--skip-svm` / `--skip-rf`, `--force-features`, `--n-jobs`.

Timing: extraction is ~25–40 images/second/core. With 400 images per class
(~15k images) expect 10–15 minutes of extraction on a Colab CPU, then a few
minutes for the RF and roughly 10–30 minutes for the full SVM grid. Start with
`--quick --max-per-class 150` to confirm it runs, then launch the real one.

## 5. What lands where

```
outputs/
  checkpoints/  a1_<arch>.pt, a2_svm.joblib, a2_rf.joblib
  features/     cached feature matrices
  figures/      confusion matrices, training curves, Grad-CAM grid,
                segmentation examples, feature-importance bars
  metrics/      one JSON per model: accuracy, balanced accuracy, macro-F1,
                weighted F1, per-class precision/recall/F1, chosen
                hyperparameters, group importances
```

Macro-F1 is the headline metric everywhere, not accuracy — PlantVillage is
imbalanced enough that accuracy flatters a model that ignores the rare
classes.

## 6. Things worth writing about in the report

These are the observations the code is set up to produce; the interpretation
and the prose are yours to write.

- **Pretrained vs scratch.** Run A1 twice (`--no-pretrained`) and compare both
  the macro-F1 and the Grad-CAM maps. The interesting claim is not just "it
  scores higher" but whether the heatmaps sit on the lesion instead of the
  background.
- **Where the two approaches get their evidence.** A2 prints impurity and
  permutation importance aggregated to colour / texture / shape. If colour
  dominates, say so — and note that PlantVillage's uniform lab backgrounds
  make colour unusually reliable here compared to field photographs.
- **Grad-CAM is a claim, not a ground truth.** It shows where the *gradient*
  is large, which is not the same as what the model needs. The A2 masks give
  you a leaf/background reference to check A1 against — that comparison is
  Phase C material.
- **Segmentation failures.** `segment_leaf` uses Excess Green, which
  under-segments heavily necrotic (brown) leaves; the saturation fallback
  catches most of it. Include a failure case in the report rather than
  claiming the segmentation is perfect.

## 7. Reproducibility

Seeds are fixed (`config.SEED = 42`) for Python, NumPy and torch, and cuDNN is
put in deterministic mode. Re-running a script reproduces the same split, the
same subsample and the same model selection. Small GPU-level nondeterminism in
float accumulation can still move the third decimal of the F1.

## 8. Not included yet

A3 (custom CNN) and A4 (ViT) are not in this drop. Both slot into the same
structure: add a builder to `approach1_transfer.py`-style script, reuse
`torch_data.make_dataloaders`, `utils.compute_metrics` and `gradcam.py`
unchanged.
