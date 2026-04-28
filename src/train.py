"""Training entry point for both PyTorch and TensorFlow.

Usage:
    python -m src.train --framework pytorch    --data-root data/raw/EuroSATallBands
    python -m src.train --framework tensorflow --data-root data/raw/EuroSATallBands
"""
from __future__ import annotations

import argparse
from pathlib import Path

from config import MODELS_DIR, RANDOM_SEED
from src.utils import set_seeds, setup_logging


def train_pytorch(data_root: Path, epochs: int, batch_size: int, lr: float) -> None:
    import mlflow
    import torch
    from torch.utils.data import DataLoader

    from src.data_loading import EuroSATDataset, collect_eurosat_files, stratified_split
    from src.models_pytorch import build_resnet50_13ch
    from src.preprocess import get_train_transforms, get_val_transforms

    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    files = collect_eurosat_files(data_root)
    train_files, val_files, _ = stratified_split(files, seed=RANDOM_SEED)
    logger.info(f"train={len(train_files)} val={len(val_files)}")

    train_loader = DataLoader(
        EuroSATDataset(train_files, transform=get_train_transforms()),
        batch_size=batch_size, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        EuroSATDataset(val_files, transform=get_val_transforms()),
        batch_size=batch_size, shuffle=False, num_workers=2,
    )

    model = build_resnet50_13ch(num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / "resnet50_13ch.pt"

    mlflow.set_experiment("geovision-pytorch")
    with mlflow.start_run():
        mlflow.log_params({
            "framework": "pytorch", "model": "resnet50_13ch",
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
        })
        best_val_acc = 0.0

        for epoch in range(epochs):
            model.train()
            train_loss_sum, train_n = 0.0, 0
            for x, y in train_loader:
                x, y = x.float().to(device), y.long().to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * x.size(0)
                train_n += x.size(0)
            train_loss = train_loss_sum / train_n

            model.eval()
            val_loss_sum, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.float().to(device), y.long().to(device)
                    logits = model(x)
                    val_loss_sum += criterion(logits, y).item() * x.size(0)
                    correct += (logits.argmax(1) == y).sum().item()
                    total += y.size(0)
            val_loss = val_loss_sum / total
            val_acc = correct / total

            logger.info(
                f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc},
                step=epoch,
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), ckpt_path)

        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_artifact(str(ckpt_path))
        logger.info(f"best val_acc={best_val_acc:.4f}  weights: {ckpt_path}")


def train_tensorflow(data_root: Path, epochs: int, batch_size: int, lr: float) -> None:
    import mlflow
    import numpy as np
    import rasterio
    import tensorflow as tf

    from config import EUROSAT_CLASSES
    from src.data_loading import collect_eurosat_files, stratified_split
    from src.models_tensorflow import build_mobilenetv2_13ch
    from src.preprocess import get_train_transforms, get_val_transforms

    logger = setup_logging()
    logger.info(f"GPUs: {tf.config.list_physical_devices('GPU')}")

    files = collect_eurosat_files(data_root)
    train_files, val_files, _ = stratified_split(files, seed=RANDOM_SEED)
    cls_idx = {c: i for i, c in enumerate(EUROSAT_CLASSES)}

    def make_ds(file_list, transform, shuffle: bool):
        def gen():
            order = np.arange(len(file_list))
            if shuffle:
                np.random.shuffle(order)
            for i in order:
                p = file_list[int(i)]
                with rasterio.open(p) as src:
                    img = src.read().astype(np.float32)
                img = np.transpose(img, (1, 2, 0))
                img = transform(image=img)["image"]
                yield img, cls_idx[p.parent.name]

        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(64, 64, 13), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(train_files, get_train_transforms(), shuffle=True)
    val_ds = make_ds(val_files, get_val_transforms(), shuffle=False)

    model = build_mobilenetv2_13ch(num_classes=10, input_size=64)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / "mobilenetv2_13ch.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt_path), save_best_only=True, monitor="val_accuracy", mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=5, monitor="val_accuracy", mode="max", restore_best_weights=True,
        ),
    ]

    mlflow.set_experiment("geovision-tensorflow")
    mlflow.tensorflow.autolog(log_models=False)
    with mlflow.start_run():
        mlflow.log_params({
            "framework": "tensorflow", "model": "mobilenetv2_13ch",
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
        })
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
        best = max(history.history["val_accuracy"])
        mlflow.log_metric("best_val_acc", best)
        mlflow.log_artifact(str(ckpt_path))
        logger = setup_logging()
        logger.info(f"best val_acc={best:.4f}  weights: {ckpt_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--framework", choices=["pytorch", "tensorflow"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()

    set_seeds()
    if args.framework == "pytorch":
        train_pytorch(args.data_root, args.epochs, args.batch_size, args.lr)
    else:
        train_tensorflow(args.data_root, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
