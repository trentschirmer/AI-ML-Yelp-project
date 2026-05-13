# Yelp Star Rating Prediction: Embeddings, Logistic Regression, and LLM Feature Extraction

A comparison of four approaches to predicting restaurant star ratings from review text, ranging from classical ML on sentence embeddings to direct LLM inference.

---

## Dataset

[Yelp Restaurant Reviews](https://www.kaggle.com/datasets/farukalam/yelp-restaurant-reviews) — approximately 20,000 restaurant reviews with star ratings (1–5), scraped from Yelp on August 15, 2017.

> ⚠️ **Note on data age:** This dataset is from 2017. It is possible that Claude Haiku has encountered some of these reviews during pre-training, which partially confounds the direct inference results (see discussion below).

**Class distribution** is heavily skewed toward 4- and 5-star reviews, which is typical of Yelp data and affects all models' behavior toward the high end of the scale.

---

## Project Structure

| Notebook | Method |
|---|---|
| `nn_model_from_embedding.ipynb` | SentenceTransformer embeddings → PyTorch neural net |
| `logistic_regression_from_embedding.ipynb` | SentenceTransformer embeddings → logistic regression |
| `claude_models.ipynb` | Claude Haiku feature extraction → logistic regression; direct Claude inference |

---

## Methods and Results

### Approach 1 — Neural Network on Sentence Embeddings

Review text was vectorized using [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384-dimensional embeddings). A three-layer feed-forward network with BatchNorm and Dropout was trained in PyTorch. Early stopping based on validation loss indicated optimal performance at approximately **6 epochs**, after which overfitting was observed.

| Metric | Value |
|---|---|
| MAE | 0.594 stars |
| Accuracy | 56% |

```
              precision    recall  f1-score   support
      1 Star       0.44      0.55      0.49       122
     2 Stars       0.27      0.55      0.37       128
     3 Stars       0.33      0.35      0.34       207
     4 Stars       0.39      0.44      0.42       445
     5 Stars       0.83      0.65      0.73      1088
    accuracy                           0.56      1990
```

---

### Approach 2 — Logistic Regression on Sentence Embeddings

Using the same `all-MiniLM-L6-v2` embeddings, a scikit-learn logistic regression classifier was trained. No validation set was needed, so the full 80/20 train/test split was used (test set twice the size of Approach 1).

| Metric | Value |
|---|---|
| MAE | 0.543 stars |
| Accuracy | 58% |

```
              precision    recall  f1-score   support
      1 Star       0.46      0.58      0.52       243
     2 Stars       0.26      0.38      0.31       256
     3 Stars       0.34      0.41      0.38       414
     4 Stars       0.41      0.48      0.44       890
     5 Stars       0.85      0.68      0.76      2177
    accuracy                           0.58      3980
```

Logistic regression slightly outperformed the neural network. Given the moderate dataset size (~20k rows), this is expected: the well-regularized convex solver has an advantage over the neural net unless the dataset is large enough to justify the added complexity.

---

### Approach 3 — Claude Feature Extraction + Logistic Regression

Rather than using fixed embeddings, Claude Haiku was used to extract four structured features from each review via API calls:

- Food quality (0.0–1.0)
- Customer service (0.0–1.0)
- Value (0.0–1.0)
- Atmosphere (0.0–1.0)

These four features were then used to train a logistic regression classifier.

| Metric | Value |
|---|---|
| MAE | 0.378 stars |
| Accuracy | 66% |

```
              precision    recall  f1-score   support
      1 Star       0.62      0.68      0.65       243
     2 Stars       0.44      0.48      0.46       256
     3 Stars       0.46      0.55      0.50       414
     4 Stars       0.46      0.46      0.46       889
     5 Stars       0.83      0.78      0.80      2176
    accuracy                           0.66      3978
```

A substantial improvement over Approaches 1 and 2 — using only 4 features versus 384. This improvement cannot be attributed to memorization: the model never sees the star labels during feature extraction, and the subsequent logistic regression is trained from scratch on those features.

---

### Approach 4 — Direct Claude Inference

Claude Haiku was asked to predict the star rating directly from the review text, with no training, using the Anthropic Batch API. Evaluated on a 1,000-review sample.

| Metric | Value |
|---|---|
| MAE | 0.312 stars |
| Accuracy | 70% |

```
              precision    recall  f1-score   support
      1 Star       0.73      0.79      0.76       168
     2 Stars       0.48      0.65      0.55       147
     3 Stars       0.80      0.41      0.54       166
     4 Stars       0.63      0.63      0.63       191
     5 Stars       0.82      0.86      0.84       328
    accuracy                           0.70      1000
```

Best overall performance. However, the evaluation sample is smaller and the dataset predates Claude's training cutoff, so some degree of memorization cannot be ruled out.

---

## Summary

| Approach | MAE | Accuracy | Notes |
|---|---|---|---|
| NN on embeddings | 0.594 | 56% | Overfits after ~6 epochs |
| Logistic regression on embeddings | 0.543 | 58% | Comparable; simpler and slightly better |
| Claude feature extraction + LR | 0.378 | 66% | Large gain; memorization ruled out |
| Direct Claude inference | 0.312 | 70% | Best, but possible memorization |

---

## Discussion

**Why do the embedding-based models perform similarly?**
Both operate on the same 384-dimensional `all-MiniLM-L6-v2` vectors, so they share the same information ceiling. The neural net has more expressive capacity but that is largely wasted at this dataset size. The similarity of results suggests the embedding itself is the binding constraint, not the classifier.

**Why does Claude feature extraction outperform 384-dimensional embeddings?**
`all-MiniLM-L6-v2` is a general-purpose sentence encoder optimized for semantic similarity, not for the specific dimensions relevant to restaurant satisfaction. Claude's four extracted features are *task-specific* by construction — they map exactly onto what a star rating reflects. Four well-chosen features can outperform hundreds of general-purpose ones. This result is not explained by memorization.

**Is the direct inference result trustworthy?**
Partially. The strong performance of direct inference is credible — large language models are genuinely good at sentiment tasks. However, this dataset is from 2017 and is publicly available on Kaggle, meaning it plausibly appears in Claude's training data. The magnitude of the advantage over feature extraction (MAE 0.31 vs. 0.38) warrants skepticism. A cleaner test would use reviews post-dating the model's training cutoff.

**On the skewed class distribution:**
All models benefit from the skew toward 4- and 5-star reviews, since predicting high ratings is easier and more frequent. The weighted-average metrics look better than the macro-average metrics across the board. For a production use case where low-star reviews matter equally, class weighting or stratified sampling would be appropriate.

---

## Practical Takeaway

When approaching a text classification task:

1. **Try direct LLM inference first.** Zero-shot performance from a capable model is often surprisingly strong and requires no labeled data or training infrastructure.
2. **If direct inference is ineffective because it is outside of the range of the LLM's expertise,** use the LLM to extract generic features that don't require special knowledge, but are related to the task, then train a lightweight classifier on those features. This approach outperformed 384-dimensional general embeddings by a wide margin here.
3. **General-purpose embeddings + classical ML** remain a solid baseline, but are outclassed when the LLM's domain understanding can be leveraged more directly.

---

## Requirements

```
torch
sentence-transformers
scikit-learn
anthropic
pandas
numpy
```

---
