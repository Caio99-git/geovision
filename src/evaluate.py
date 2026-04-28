"""Evaluation: confusion matrix, per-class report, ROC-AUC. Outputs to data/metrics/."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from config import DATA_METRICS, EUROSAT_CLASSES, RANDOM_SEED
from src.utils import set_seeds, setup_logging

logger = setup_logging()


def evaluate_pytorch(data_root: Path, ckpt_path: Path) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from src.data_loading import EuroSATDataset, collect_eurosat_files, stratified_split
    from src.models_pytorch import build_resnet50_13ch
    from src.preprocess import get_val_transforms

    files = collect_eurosat_files(data_root)
    _, _, test_files = stratified_split(files, seed=RANDOM_SEED)
    loader = DataLoader(
        EuroSATDataset(test_files, transform=get_val_transforms()),
        batch_size=64, shuffle=False, num_workers=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet50_13ch(num_classes=10).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    ys, probs = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(model(x.float().to(device)), dim=1).cpu().numpy()
            ys.append(y.numpy())
            probs.append(p)
    return _save(np.concatenate(ys), np.concatenate(probs), framework="pytorch")


def evaluate_tensorflow(data_root: Path, ckpt_path: Path) -> dict:
    import rasterio
    import tensorflow as tf

    from src.data_loading import collect_eurosat_files, stratified_split
    from src.preprocess import get_val_transforms

    cls_idx = {c: i for i, c in enumerate(EUROSAT_CLASSES)}
    files = collect_eurosat_files(data_root)
    _, _, test_files = stratified_split(files, seed=RANDOM_SEED)
    transform = get_val_transforms()

    model = tf.keras.models.load_model(ckpt_path)

    ys, probs, batch_x, batch_y = [], [], [], []

    def flush():
        if batch_x:
            probs.append(model.predict(np.stack(batch_x), verbose=0))
            ys.extend(batch_y)
            batch_x.clear()
            batch_y.clear()

    for p in test_files:
        with rasterio.open(p) as src:
            img = src.read().astype(np.float32)
        img = transform(image=np.transpose(img, (1, 2, 0)))["image"]
        batch_x.append(img)
        batch_y.append(cls_idx[p.parent.name])
        if len(batch_x) == 64:
            flush()
    flush()

    return _save(np.array(ys), np.concatenate(probs, axis=0), framework="tensorflow")


def _save(y_true: np.ndarray, probs: np.ndarray, *, framework: str) -> dict:
    DATA_METRICS.mkdir(parents=True, exist_ok=True)
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=EUROSAT_CLASSES, output_dict=True)
    auc_macro = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")

    pd.DataFrame(cm, index=EUROSAT_CLASSES, columns=EUROSAT_CLASSES).to_csv(
        DATA_METRICS / f"confusion_matrix_{framework}.csv"
    )
    pd.DataFrame(report).T.to_csv(DATA_METRICS / f"classification_report_{framework}.csv")
    pd.DataFrame([{"framework": framework, "accuracy": acc, "auc_macro": auc_macro}]).to_csv(
        DATA_METRICS / f"summary_{framework}.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), EUROSAT_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), EUROSAT_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {framework} (acc={acc:.3f}, AUC={auc_macro:.3f})")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(DATA_METRICS / f"confusion_matrix_{framework}.png", dpi=150)
    plt.close(fig)

    logger.info(f"{framework}: acc={acc:.4f}  AUC={auc_macro:.4f}")
    return {"framework": framework, "accuracy": acc, "auc_macro": auc_macro}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--framework", choices=["pytorch", "tensorflow"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    args = p.parse_args()

    set_seeds()
    if args.framework == "pytorch":
        evaluate_pytorch(args.data_root, args.ckpt)
    else:
        evaluate_tensorflow(args.data_root, args.ckpt)


if __name__ == "__main__":
    main()
