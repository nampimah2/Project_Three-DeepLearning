# IMDb Sentiment Classification with BiGRU and BiLSTM

**Nicholas Ampimah · University of Nebraska–Lincoln**

This project classifies IMDb movie reviews as positive or negative. It compares a bidirectional GRU and a bidirectional LSTM, each with masked attention pooling and two classifier dropout rates: **0.20 (A)** and **0.50 (B)**.

## Data and models

The code loads `imdb_reviews` through TensorFlow Datasets. The official training split is stratified into 20,000 training and 5,000 validation reviews (seed 42). The vocabulary is fitted on training reviews only, with up to 15,000 entries and sequences padded or truncated to 200 tokens. The 25,000 official test reviews are held out until model selection is complete.

Both encoders use 64-dimensional embeddings, 48 recurrent units per direction, and attention pooling that excludes padding. The classifier uses a 32-unit ReLU layer, dropout, and a sigmoid output. Training uses Adam (learning rate 0.001), binary cross-entropy, and early stopping on validation loss with patience 5.

| File | Purpose |
| --- | --- |
| `main.py` | Train four configurations, select by validation loss, and evaluate on the held-out test set |
| `model.py` | Define the recurrent encoders, masked attention pooling, and classifier |
| `util.py` | Load and preprocess data, calculate metrics and confidence intervals, and generate figures |

## Run the experiment

Use an environment with TensorFlow, TensorFlow Datasets, NumPy, scikit-learn, and Matplotlib. On Swan, run from the directory containing the three scripts in a compute allocation with at least four CPU cores. If using the existing `cifar100` environment:

```bash
conda activate cifar100   # Your environment with the require packages
python -u main.py --output-dir results_run3 --data-dir tfds_cache \
  --epochs 300 --patience 5 --batch-size 128 --threads 4
```

Use a new or empty output directory for each run. Keep the prepared `tfds_cache/` to reuse the dataset. If it is not already prepared and the compute node cannot reach Stanford, prepare the TensorFlow Datasets cache separately using the original IMDb archive and put it on shared work storage. The epoch count is a cap; all four configurations in the reported run stopped after five epochs without validation loss improvement.

## Test results

| Model | Best / stop epoch | Validation loss | Test accuracy | Macro-F1 | Test error (95% Wilson CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| BiGRU-A | 2 / 7 | 0.3040 | 85.06% | 0.850 | 14.94% (14.51–15.39%) |
| BiGRU-B | 2 / 7 | 0.3055 | 85.10% | 0.851 | 14.90% (14.47–15.35%) |
| **BiLSTM-A (selected)** | **2 / 7** | **0.3027** | **84.84%** | **0.848** | **15.16% (14.72–15.61%)** |
| BiLSTM-B | 1 / 6 | 0.3179 | 85.15% | 0.851 | 14.85% (14.41–15.29%) |

BiLSTM-A was selected **before testing** because it had the lowest validation loss. BiLSTM-B had the highest observed test accuracy, but the test scores were used only for comparison. Increasing dropout to 0.50 did not improve minimum validation loss for either encoder. All four learning curves showed overfitting after the best checkpoint. These results are from one seed.

## Generated outputs

- `metrics.csv`: test accuracy, macro-F1, ROC AUC, error, and Wilson error intervals.
- `selection.json` and `run_config.json`: validation selection and run settings.
- `confusion_counts.json` and `confusion_matrices.pdf`: raw counts and row-normalized confusion matrices for all four models.
- `*_epochs.csv` and `learning_curves.pdf`: per-epoch histories and training/validation loss curves.
- `*.weights.h5`, `*_test_predictions.npz`, and `vocabulary.json`: model checkpoints, test probabilities and labels, and the fitted vocabulary.

The three-page IEEE report is prepared separately. The scripts produce experimental outputs only.

Dataset reference: A. L. Maas et al., *Learning Word Vectors for Sentiment Analysis*, ACL-HLT, 2011.
