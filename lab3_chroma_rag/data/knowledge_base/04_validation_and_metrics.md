# Model Validation Methodologies, Statistical Evaluation, and Metric Selection

## 1. Cross-Validation and Data Splitting Protocols
Rigorous empirical evaluation protects production models from optimistic bias and unseen distribution shifts.

### 1.1 Partitioning Strategies
- **Stratified K-Fold**: Maintains target class balance across folds; critical for severely imbalanced classification tasks ($< 5\%$ positive rate).
- **Temporal Block Splitting (Purged & Embargoed Cross-Validation)**: Eliminates auto-correlation and information leakage in time-series data. Training periods strictly precede validation periods, with buffer intervals (embargoes) removing post-event overlap.
- **GroupKFold**: Guarantees that records from the same real-world entity (e.g., patient, sensor cluster, company) reside exclusively in train or validation sets, testing generalization across distinct entities.

## 2. Comprehensive Metric Formulations
Evaluation metrics must mirror business utility and loss asymmetries:

### 2.1 Classification Under Asymmetric Costs
- **Precision, Recall, and $F_\beta$-Score**:
$$F_\beta = (1 + \beta^2) \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$
When False Negatives carry prohibitive costs (e.g., medical diagnostics, fraud detection), setting $\beta = 2$ weights recall over precision.
- **PR-AUC vs. ROC-AUC**: In heavily skewed datasets, ROC-AUC yields overly optimistic interpretations due to massive True Negative counts. Precision-Recall AUC (PR-AUC) evaluates trade-offs exclusively on the minority class.
- **Brier Score and Calibration**: Assesses probabilistic reliability:
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
Calibrated probabilities (via Platt Scaling or Isotonic Regression) are mandatory when decisions depend on risk thresholds.

### 2.2 Ranking and Retrieval Metrics
Information retrieval and recommendation engines rely on position-aware quality metrics:
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of the first relevant retrieved document:
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
- **Normalized Discounted Cumulative Gain (NDCG@K)**: Measures graded relevance with logarithmic rank discounting:
$$\text{DCG}@K = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

## 3. Offline-to-Online Evaluation Parity
Offline validation gains frequently fail to translate to online business impact due to feedback loops, cannibalization, or delayed label attribution. Establishing explicit offline-online correlation benchmarks is essential prior to deployment.
