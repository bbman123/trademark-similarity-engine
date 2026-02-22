# step1_2.ipynb — Pipeline Guide

This notebook implements the full end-to-end training pipeline for the Trademark Similarity Engine in a single file, covering data preprocessing (Step 1), CNN encoder training (Step 2), and hybrid SVM classifier training (Step 3).

---

## Notebook Structure

| Cell | Description |
|------|-------------|
| **Cell 1** | Environment setup — sets HuggingFace env vars, detects PyTorch device |
| **Cell 2** | Sanity check — prints PyTorch version and SentenceTransformer import |
| **Cell 3** | **Step 1**: Full preprocessing and feature engineering pipeline |
| **Cell 4** | **Steps 2 & 3**: CNN training + Hybrid CNN+SVM training + evaluation |

> Cell 3 (the pip install cell) is intentionally left unexecuted to avoid overwriting the pinned package versions. See [requirements.txt](requirements.txt) for the correct install commands.

---

## Step 1 — Data Preprocessing (Cell 3)

### Input

| File | Path | Description |
|------|------|-------------|
| Raw trademark data | `data/trademark_file.csv` | Source CSV with 18,737 trademark opposition pairs (Portuguese/multilingual) |

**Input CSV columns used:**

| Column | Renamed to | Description |
|--------|------------|-------------|
| `Process number RM` | `mark1_id` | Trademark 1 registration number |
| `Process number TM` | `mark2_id` | Trademark 2 registration number |
| `Name RM` | `mark1_wordmark` | Trademark 1 text |
| `Name TM` | `mark2_wordmark` | Trademark 2 text |
| `Status RM` | `mark1_status` | Legal status of trademark 1 |
| `Status TM` | `mark2_status` | Legal status of trademark 2 |

### Processing Stages

```
trademark_file.csv (18,737 rows)
        │
        ▼
1. Load & encoding detection (MacRoman/utf-8/latin-1 fallback)
        │
        ▼
2. Column extraction + label = 1 (all are similar pairs)
        │
        ▼
3. Remove invalid wordmarks  → -381 rows
        │
        ▼
4. Deduplicate pairs         → -3,553 rows
        │
        ▼
5. Translation pipeline
   ├── Portuguese → English  (Google Translate, with fallback)
   ├── English → Hausa       (domain lexicon + Google Translate)
   └── English → Yoruba      (domain lexicon + Google Translate)
        │
        ▼
6. Synthetic negative pair generation (ratio = 1.0)
   → 14,803 dissimilar pairs created (length-difference strategy)
        │
        ▼
7. Feature extraction
   ├── Visual:    Levenshtein distance, Jaro-Winkler similarity
   ├── Phonetic:  Soundex match, Metaphone match (on English text)
   └── Semantic:  Cosine similarity of sentence embeddings (EN, HA, YO)
        │
        ▼
Output: 29,606 pairs × 24 features
```

### Feature Columns in Output CSV

| Feature | Type | Description |
|---------|------|-------------|
| `mark1_wordmark` | str | Original trademark 1 text |
| `mark2_wordmark` | str | Original trademark 2 text |
| `mark1_wordmark_en` | str | English translation of mark 1 |
| `mark2_wordmark_en` | str | English translation of mark 2 |
| `mark1_wordmark_ha` | str | Hausa translation of mark 1 |
| `mark2_wordmark_ha` | str | Hausa translation of mark 2 |
| `mark1_wordmark_yo` | str | Yoruba translation of mark 1 |
| `mark2_wordmark_yo` | str | Yoruba translation of mark 2 |
| `label` | int | 1 = similar, 0 = dissimilar |
| `pair_type` | str | `positive` or `negative` |
| `mark1_status` | str | Legal status (positive pairs only) |
| `mark2_status` | str | Legal status (positive pairs only) |
| `mark1_id` | str | Registration ID (positive pairs only) |
| `mark2_id` | str | Registration ID (positive pairs only) |
| `mark1_length` | int | Character count of mark 1 |
| `mark2_length` | int | Character count of mark 2 |
| `length_diff` | int | Absolute length difference |
| `visual_levenshtein` | int | Edit distance between wordmarks |
| `visual_jaro_winkler` | float | Jaro-Winkler string similarity (0–1) |
| `soundex_match` | int | 1 if Soundex codes match, else 0 |
| `metaphone_match` | int | 1 if Metaphone codes match, else 0 |
| `semantic_similarity_en` | float | Cosine similarity of English embeddings |
| `semantic_similarity_ha` | float | Cosine similarity of Hausa embeddings |
| `semantic_similarity_yo` | float | Cosine similarity of Yoruba embeddings |

### Output Files

| File | Description |
|------|-------------|
| `data/trademark_similarity_dataset_final.csv` | Primary output (auto-incremented to `_1.csv`, `_2.csv`… if already exists) |
| `eda_visualizations_YYYYMMDD_HHMMSS/` | Directory with 7 PNG visualisation charts |
| `pipeline_config.json` | Stores the path of the most recently generated CSV for Step 2 to pick up automatically |

**Most recent run output:** `data/trademark_similarity_dataset_final_7.csv` (29,606 rows × 24 columns, 8.44 MB)

### EDA Visualisations

Generated in `eda_visualizations_YYYYMMDD_HHMMSS/`:

| File | Contents |
|------|----------|
| `01_label_distribution.png` | Pie chart — similar vs dissimilar split |
| `02_length_distribution.png` | Histogram of wordmark character lengths |
| `03_visual_similarity.png` | Levenshtein + Jaro-Winkler distributions by label |
| `04_phonetic_matches.png` | Soundex & Metaphone match counts by label |
| `05_semantic_similarity.png` | Cosine similarity distributions across EN/HA/YO |
| `06_correlation_heatmap.png` | Feature correlation matrix |
| `07_length_difference.png` | Length difference box plot by label |

**Last run stats:**
- 29,606 total pairs (50% similar / 50% dissimilar)
- Average wordmark length: 16.2 characters
- Jaro-Winkler mean: 0.496
- Soundex match rate: 12.6%
- Semantic similarity (EN): 0.446

---

## Step 2 — CNN Encoder Training (Cell 4, first half)

### Input

`pipeline_config.json` → points to the CSV from Step 1 (e.g. `data/trademark_similarity_dataset_final_7.csv`)

### Architecture

A **Siamese Character-level CNN** learns to embed trademark strings into 64-dimensional vectors. Both marks share the same encoder weights.

```
Input text (padded char sequence, length=50)
        │
        ▼
Character Embedding (vocab ≈ 54 chars → 128 dims)
        │
        ├── Conv1D(128 filters, kernel=3) → GlobalMaxPool
        ├── Conv1D(64  filters, kernel=4) → GlobalMaxPool
        └── Conv1D(32  filters, kernel=5) → GlobalMaxPool
                        │
                        ▼
              Concatenate [128+64+32 = 224 dims]
                        │
                        ▼
              Dense(256) → Dropout(0.5)
              Dense(128) → Dropout(0.5)
              Dense(64)  ← encoder output
```

The Siamese model computes `|emb1 - emb2|`, `emb1 * emb2`, then feeds `[emb1, emb2, diff, product]` (256 dims) into a classification head with sigmoid output.

### Training Config

| Parameter | Value |
|-----------|-------|
| Max sequence length | 50 chars |
| Embedding dim | 128 |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping patience | 10 |
| Optimiser | Adam (lr=0.001, ReduceLROnPlateau) |
| Loss | Binary cross-entropy |

### Data Splits

| Split | Size | Notes |
|-------|------|-------|
| Train | 20,724 (70%) | Stratified |
| Validation | 4,441 (15%) | Stratified |
| Test | 4,441 (15%) | Stratified |

### Output Files — CNN

| File | Description |
|------|-------------|
| `models/best_cnn_model.h5` | Best checkpoint during training (by val_loss) |
| `models/cnn_encoder.keras` | Full Siamese CNN model (Keras format) |
| `models/cnn_encoder_tokenizer.pkl` | Character tokenizer (needed for inference) |
| `results/cnn_training_history.png` | Loss + accuracy curves |

**Last run:** Training stopped at epoch 28 (early stopping). Val accuracy peaked at ~77%.

---

## Step 3 — Hybrid CNN+SVM Classifier (Cell 4, second half)

### Input

- CNN encoder from Step 2 (extracts 64-dim embeddings per mark)
- Pre-computed linguistic features from Step 1 (10 features)

### Feature Vector

For each pair, the SVM receives a **138-dimensional** feature vector:

```
[emb1 (64) | emb2 (64) | linguistic_features (10)] = 138 dims
```

Linguistic features used:
`visual_levenshtein`, `visual_jaro_winkler`, `soundex_match`, `metaphone_match`, `semantic_similarity_en`, `semantic_similarity_ha`, `semantic_similarity_yo`, `length_diff`, `mark1_length`, `mark2_length`

All features are `StandardScaler`-normalised before SVM training.

### SVM Config

| Parameter | Value |
|-----------|-------|
| Kernel | RBF |
| C | 1.0 |
| Gamma | scale |
| Probability | True |

### Results (last run — 29,606 pairs)

| Metric | Value |
|--------|-------|
| Training accuracy | 89.09% |
| Validation accuracy | 85.81% |
| **Test accuracy** | **86.49%** |
| **Precision** | **89.71%** |
| **Recall** | **82.43%** |
| **F1 Score** | **85.92%** |
| **ROC-AUC** | **93.49%** |

Confusion matrix (test set, 4,441 pairs):

|  | Predicted Dissimilar | Predicted Similar |
|--|----------------------|-------------------|
| **True Dissimilar** | 2,011 | 210 |
| **True Similar** | 390 | 1,830 |

### Output Files — SVM + Results

| File | Description |
|------|-------------|
| `models/hybrid_svm.pkl` | Trained SVM + fitted StandardScaler |
| `results/confusion_matrix.png` | Confusion matrix heatmap |
| `results/roc_curve.png` | ROC curve (AUC = 0.935) |
| `results/evaluation_results.json` | All metrics + data split sizes as JSON |

---

## Device Configuration

| Framework | Device | Notes |
|-----------|--------|-------|
| PyTorch / SentenceTransformer | CUDA (RTX 4070) | GPU accelerated |
| TensorFlow / CNN | CPU (`/CPU:0`) | TF dropped native Windows CUDA after v2.10 |
| SVM (scikit-learn) | CPU | Standard |

---

## Re-running the Pipeline

To re-run from scratch:

```bash
# Activate environment
.venv\Scripts\activate

# Open notebook and run cells in order: 1 → 3 → 4
# (Cell 2 is optional; Cell 3/pip cell should NOT be re-run)
```

Each run of Cell 3 auto-increments the output CSV filename (`_1`, `_2`, ...) to avoid overwriting previous results. `pipeline_config.json` is updated automatically so Cell 4 always reads the freshest file.

To use a specific dataset for Cell 4, edit `pipeline_config.json`:
```json
{
  "processed_data_file": "data/trademark_similarity_dataset_final_7.csv",
  "timestamp": "2026-02-22T..."
}
```
