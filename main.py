"""Train four IMDb sentiment classifiers; select on validation before testing."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from model import build_model
from util import load_training_data, load_test_data, batches, evaluate, make_figures

CONFIGS = [(arch, setting, rate) for arch in ("GRU", "LSTM")
           for setting, rate in (("A", 0.20), ("B", 0.50))]


def set_seed(seed):
    """Set the random seeds used by Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--data-dir", default=None, help="TensorFlow Datasets cache directory")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--epochs", type=int, default=300,
                   help="Safety cap; a run hitting it without full patience is rejected")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    if args.patience < 5 or args.epochs <= args.patience or min(args.batch_size, args.threads) < 1:
        p.error("Require patience >= 5, epochs > patience, and positive batch size and threads.")

    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, args.threads // 2))

    out = Path(args.output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        p.error("Choose an empty output directory to avoid mixing results from different runs.")
    out.mkdir(parents=True, exist_ok=True)

    seed = 42
    set_seed(seed)

    (out / "run_config.json").write_text(
        json.dumps({
            **vars(args),
            "seed": seed,
            "tensorflow": tf.__version__,
            "numpy": np.__version__
        }, indent=2),
        encoding="utf-8"
    )

    tr, va, vectorizer = load_training_data(args.data_dir, seed, args.threads)
    length = vectorizer.get_config()["output_sequence_length"]
    vocab = vectorizer.get_vocabulary()

    (out / "vocabulary.json").write_text(
        json.dumps(vocab),
        encoding="utf-8"
    )

    runs, histories = [], {}

    for arch, setting, dropout in CONFIGS:
        tf.keras.backend.clear_session()
        set_seed(seed)

        name = f"Bi{arch}-{setting}"
        model = build_model(arch, len(vocab), length, dropout)

        train = batches(tr, args.batch_size, args.threads, seed=seed)
        valid = batches(va, args.batch_size, args.threads)

        early = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=0.0,
            patience=args.patience,
            restore_best_weights=True
        )

        checkpoint = out / f"{name}.weights.h5"

        print(f"\n=== {name} (dropout={dropout:.2f}) ===", flush=True)

        hist = model.fit(
            train,
            validation_data=valid,
            epochs=args.epochs,
            verbose=2,
            callbacks=[
                early,
                tf.keras.callbacks.ModelCheckpoint(
                    str(checkpoint),
                    monitor="val_loss",
                    mode="min",
                    save_best_only=True,
                    save_weights_only=True
                ),
                tf.keras.callbacks.CSVLogger(
                    str(out / f"{name}_epochs.csv")
                )
            ]
        ).history

        best = int(np.argmin(hist["val_loss"]))

        if len(hist["val_loss"]) - best - 1 < args.patience:
            raise RuntimeError(
                f"{name} hit the epoch cap before {args.patience} epochs "
                "without improvement. Increase --epochs and use a fresh output directory."
            )

        histories[name] = hist

        runs.append(dict(
            model=name,
            architecture=arch,
            dropout=dropout,
            params=model.count_params(),
            epochs=len(hist["val_loss"]),
            best_epoch=best + 1,
            val_loss=float(hist["val_loss"][best]),
            val_accuracy=float(hist["val_accuracy"][best]),
            train_accuracy=float(hist["accuracy"][best]),
            train_loss=float(hist["loss"][best])
        ))

    # Freeze model selection before loading or evaluating on the official test set.
    chosen = min(runs, key=lambda row: row["val_loss"])["model"]

    (out / "selection.json").write_text(
        json.dumps({
            "selected": chosen,
            "criterion": "minimum validation binary cross-entropy",
            "candidates": runs
        }, indent=2),
        encoding="utf-8"
    )

    print(f"Selected before testing: {chosen}", flush=True)

    test = load_test_data(args.data_dir, vectorizer, args.threads)
    x, y = test

    matrices = {}
    pred_ds = batches(x, args.batch_size, args.threads)

    for row in runs:
        model = build_model(
            row["architecture"],
            len(vocab),
            length,
            row["dropout"]
        )

        model.load_weights(
            str(out / f'{row["model"]}.weights.h5')
        )

        probs = model.predict(
            pred_ds,
            verbose=0
        ).reshape(-1)

        metrics, counts = evaluate(y, probs)

        row.update(
            metrics,
            selected=(row["model"] == chosen)
        )

        matrices[row["model"]] = counts.tolist()

        np.savez_compressed(
            out / f'{row["model"]}_test_predictions.npz',
            labels=y,
            probabilities=probs
        )

    with (out / "metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(runs[0])
        )
        w.writeheader()
        w.writerows(runs)

    (out / "confusion_counts.json").write_text(
        json.dumps(matrices, indent=2),
        encoding="utf-8"
    )

    make_figures(
        out,
        runs,
        histories,
        matrices
    )

    print(
        f"Results saved to {out.resolve()}",
        flush=True
    )


if __name__ == "__main__":
    main()
