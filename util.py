"""Data preparation, held-out evaluation, and figure generation."""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

MAX_TOKENS = 15000
MAX_LENGTH = 200


def _options(threads):
    opts = tf.data.Options()
    opts.threading.private_threadpool_size = threads
    opts.threading.max_intra_op_parallelism = 1
    return opts


def _read_split(split, data_dir, threads):
    ds = tfds.load("imdb_reviews", split=split, as_supervised=True,
                   data_dir=data_dir, shuffle_files=False)
    examples = list(tfds.as_numpy(ds.with_options(_options(threads))))
    texts, labels = zip(*examples)
    return np.asarray(texts, dtype=object), np.asarray(labels, dtype=np.int32)


@tf.keras.utils.register_keras_serializable(package="IMDb")
def clean_text(s):
    s = tf.strings.lower(s)
    s = tf.strings.regex_replace(s, r"<br\s*/?>", " ")
    return tf.strings.regex_replace(s, r"[^a-z0-9'\s]+", " ")


def _encode(vectorizer, texts):
    return np.concatenate([vectorizer(tf.constant(texts[i:i+512].tolist())).numpy()
                           for i in range(0, len(texts), 512)]).astype(np.int32)


def load_training_data(data_dir, seed, threads):
    texts, y = _read_split("train", data_dir, threads)
    train, val = train_test_split(np.arange(len(y)), test_size=0.20, stratify=y,
                                  random_state=seed)
    vec = tf.keras.layers.TextVectorization(max_tokens=MAX_TOKENS, standardize=clean_text,
        output_mode="int", output_sequence_length=MAX_LENGTH)
    adapt_ds = tf.data.Dataset.from_tensor_slices(tf.constant(texts[train].tolist()))
    vec.adapt(adapt_ds.batch(256).with_options(_options(threads)))
    return (_encode(vec, texts[train]), y[train]), (_encode(vec, texts[val]), y[val]), vec


def load_test_data(data_dir, vectorizer, threads):
    texts, y = _read_split("test", data_dir, threads)
    return _encode(vectorizer, texts), y


def batches(data, batch_size, threads, seed=None):
    ds = tf.data.Dataset.from_tensor_slices(data)
    if seed is not None:
        ds = ds.shuffle(len(data[0]), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(1).with_options(_options(threads))


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    delta = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return max(0.0, center-delta), min(1.0, center+delta)


def evaluate(y, p):
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, log_loss
    predicted = (p > 0.5).astype(int)
    count = int(np.count_nonzero(predicted != y))
    low, high = wilson(count, len(y))
    return dict(n_test=len(y), accuracy=1-count/len(y), error=count/len(y),
                error_ci_low=low, error_ci_high=high, macro_f1=f1_score(y, predicted, average="macro"),
                auc=roc_auc_score(y, p), test_loss=log_loss(y, p)), confusion_matrix(y, predicted, labels=[0, 1])


def make_figures(out, runs, histories, matrices):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(7, 2.0), layout="constrained")
    for ax, row in zip(axes, runs):
        raw = np.asarray(matrices[row["model"]])
        norm = raw / raw.sum(axis=1, keepdims=True)
        ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for i,j in np.ndindex(2,2):
            ax.text(j, i, f"{norm[i,j]:.1%}", ha="center", va="center",
                    color="white" if norm[i,j] > 0.6 else "#153350", fontsize=9)
        ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["Neg.", "Pos."],
               yticklabels=["Neg.", "Pos."], title=row["model"], xlabel="Predicted")
        if ax is axes[0]:
            ax.set_ylabel("True")
    fig.savefig(out / "confusion_matrices.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(8, 5), layout="constrained")
    for ax, row in zip(axes.flat, runs):
        h = histories[row["model"]]
        e = np.arange(1, 1+len(h["loss"]))
        ax.plot(e, h["loss"], label="Training")
        ax.plot(e, h["val_loss"], label="Validation")
        ax.axvline(row["best_epoch"], color="gray", ls=":", label="Best epoch")
        ax.set(title=row["model"], xlabel="Epoch", ylabel="Cross-entropy")
        ax.legend(fontsize=8)
    fig.savefig(out / "learning_curves.pdf")
    plt.close(fig)
