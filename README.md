# IMDb Sentiment Classification with BiGRU and BiLSTM

**Nicholas Ampimah, Yin Po Po Aung, and Jane Ohia · University of Nebraska–Lincoln**

This project classifies IMDb movie reviews as positive or negative. It compares a bidirectional GRU and a bidirectional LSTM, each with masked attention pooling and two classifier dropout rates: **0.20 (A)** and **0.50 (B)**.

## Data and models

The code loads `imdb_reviews` through TensorFlow Datasets. The official training split is stratified into 20,000 training and 5,000 validation reviews (seed 42). The vocabulary is fitted on training reviews only, with up to 15,000 entries and sequences padded or truncated to 200 tokens. The 25,000 official test reviews are held out until model selection is complete.

Both encoders use 64-dimensional embeddings, 48 recurrent units per direction, and attention pooling that excludes padding. The classifier uses a 32-unit ReLU layer, dropout, and a sigmoid output. Training uses Adam (learning rate 0.001), binary cross-entropy, and early stopping on validation loss with patience 5.

| File | Purpose |
| --- | --- |
| `main.py` | Train four configurations, select by validation loss, and evaluate on the held-out test set |
| `model.py` | Define the recurrent encoders, masked attention pooling, and classifier |
| `util.py` | Load and preprocess data, calculate metrics, confidence intervals, and paired McNemar comparisons, and generate figures |

## Dataset setup

The code reads `imdb_reviews` from a local TensorFlow Datasets cache folder named `tfds_cache/` in the project directory. Prepare it with one of these options.

**Option 1: download it (needs internet access)**

Run this once from the project directory, on a machine or login node that can reach the internet:

```bash
python -c "import tensorflow_datasets as tfds; tfds.load('imdb_reviews', split=['train','test'], data_dir='tfds_cache')"
ls tfds_cache
```

The folder should now contain `imdb_reviews` (and a `downloads` folder). Compute nodes on a cluster often have no internet access, so download on the login node first, then run the training job.

**Option 2: use a shared cache**

If a teammate shares a `tfds_cache.zip`, extract it so that `downloads` and `imdb_reviews` sit directly inside `tfds_cache/`:

```bash
mkdir -p tfds_cache
unzip tfds_cache.zip -d tfds_cache
ls tfds_cache
```

If `ls` shows a single extra `tfds_cache` folder inside, move its contents up one level.

The cache is large, so it is not stored in this repository (it is listed in `.gitignore`).

## Run the experiment

Use an environment with TensorFlow, TensorFlow Datasets, NumPy, SciPy, scikit-learn, and Matplotlib. On Swan, run from the directory containing the three scripts in a compute allocation with at least four CPU cores. If using the existing `cifar100` environment:

```bash
conda activate cifar100   # Your environment with the require packages
python -u main.py --output-dir results_run4 --data-dir tfds_cache \
  --epochs 300 --patience 5 --batch-size 128 --threads 4
```

Use a new or empty output directory for each run. Keep the prepared `tfds_cache/` to reuse the dataset. If it is not already prepared and the compute node cannot reach Stanford, prepare the TensorFlow Datasets cache separately using the original IMDb archive and put it on shared work storage. The epoch count is a cap; all four configurations in the reported run stopped after five epochs without validation loss improvement.

To recompute the paired model comparisons from saved predictions without training:

```bash
python main.py --output-dir results_run4 --compare-only
```

## Test results

| Model | Best / stop epoch | Validation loss | Test accuracy | Macro-F1 | Test error (95% Wilson CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| BiGRU-A | 2 / 7 | 0.3040 | 85.06% | 0.850 | 14.94% (14.51–15.39%) |
| BiGRU-B | 2 / 7 | 0.3055 | 85.10% | 0.851 | 14.90% (14.47–15.35%) |
| **BiLSTM-A (selected)** | **2 / 7** | **0.3038** | **84.94%** | **0.849** | **15.06% (14.63–15.51%)** |
| BiLSTM-B | 2 / 7 | 0.3170 | 84.11% | 0.841 | 15.89% (15.44–16.35%) |

BiLSTM-A was selected **before testing** because it had the lowest validation loss. BiGRU-B had the highest observed test accuracy, but the test scores were used only for comparison. Increasing dropout to 0.50 did not improve minimum validation loss for either encoder. All four learning curves showed overfitting after the best checkpoint. These results are from one seed; an earlier run of the same code gave identical BiGRU results but different BiLSTM results.

## Paired comparison (McNemar exact test, same 25,000 test reviews)

| Pair | Accuracy difference | 95% CI | p |
| --- | ---: | --- | ---: |
| BiGRU-A vs BiGRU-B | −0.04 pts | [−0.29, +0.21] | 0.77 |
| BiGRU-A vs BiLSTM-A | +0.12 pts | [−0.14, +0.38] | 0.37 |
| BiGRU-A vs BiLSTM-B | +0.94 pts | [+0.61, +1.27] | <0.0001 |
| BiGRU-B vs BiLSTM-A | +0.16 pts | [−0.12, +0.44] | 0.28 |
| BiGRU-B vs BiLSTM-B | +0.98 pts | [+0.66, +1.31] | <0.0001 |
| BiLSTM-A vs BiLSTM-B | +0.82 pts | [+0.49, +1.16] | <0.0001 |

BiGRU-A, BiGRU-B, and BiLSTM-A are not significantly different from one another, while BiLSTM-B is significantly worse than each of them.

## Generated outputs

- `metrics.csv`: test accuracy, macro-F1, ROC AUC, error, and Wilson error intervals.
- `selection.json` and `run_config.json`: validation selection and run settings.
- `confusion_counts.json` and `confusion_matrices.pdf`: raw counts and row-normalized confusion matrices for all four models.
- `*_epochs.csv` and `learning_curves.pdf`: per-epoch histories and training/validation loss curves.
- `*.weights.h5`, `*_test_predictions.npz`, and `vocabulary.json`: model checkpoints, test probabilities and labels, and the fitted vocabulary.

The four-page IEEE report is prepared separately. The scripts produce experimental outputs only.

Dataset reference: A. L. Maas et al., *Learning Word Vectors for Sentiment Analysis*, ACL-HLT, 2011.