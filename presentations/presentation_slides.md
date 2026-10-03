# 📊 10-Minute Oral Defense Slide Deck
**Project:** End-to-End Data Preparation, Clustering & Classification  
**Dataset:** Adult Census Income (UCI Machine Learning Repository)  
**Course:** ESPRIT — 4CCE9 (2026–2027)  
**Instructor:** Mohamed Aziz KASSEB  
**Author(s):** [Your Name / Team Members]  

---

## Slide 1: Title & Context
- **Title:** Socioeconomic Income Level Prediction & Data Preparation Pipeline
- **Methodological Framework:** CRISP-DM (Cross-Industry Standard Process for Data Mining)
- **Dataset:** 1994 U.S. Census Bureau database (48,842 observations across 15 attributes)
- **Objective:** Predict binary target `income` (`<=50K` vs `>50K`) based on demographic, occupational, and educational features.

---

## Slide 2: Business Understanding & Problem Framing
- **Strategic Utility:**
  - Automated loan & credit pre-qualification (mitigating invasive financial audits).
  - Targeted wealth management & financial advisory offerings.
  - Policy analysis on socioeconomic wage determinants.
- **Asymmetric Cost of Errors:**
  - **False Positive (FP):** Predicting high income when applicant earns $\le$50K $\rightarrow$ **High default risk** on loans.
  - **False Negative (FN):** Predicting low income when applicant earns >50K $\rightarrow$ Minor lost opportunity.
- **Metric Selection:**
  - Due to class imbalance (~76% $\le$50K vs ~24% >50K), **Accuracy is deceptive**.
  - Primary evaluation relies on **ROC-AUC, Precision-Recall AUC (PR-AUC), and F1-Score**.

---

## Slide 3: Key Observations from Data Exploration (EDA)
- **Severe Class Imbalance:** 75.92% majority class vs 24.08% minority class.
- **Age Dynamics:**
  - Low earners peak at age 23 (early career/entry level).
  - High earners peak between ages 38 and 50 (prime career phase).
- **Education Threshold Effect:**
  - Earning probability jumps from <15% for $\le$12 years of education to >50% for Bachelor's, Master's, and Doctorate degrees (`education_num` $\ge$ 13).
- **Working Hours:** Standard week is 40 hours, but heavy overtime (>50 hours) correlates strongly with high income.
- **Zero-Inflation in Capital Accounts:** Over 91.7% have 0 capital gains, and 95.3% have 0 capital losses.

---

## Slide 4: Data Quality Issues & Hidden Traps Identified
1. **Target Inconsistency:** UCI test set appended a trailing dot `.` (`<=50K.` vs `<=50K`), creating 4 classes instead of 2.
2. **Hidden Missing Values (`?`):**
   - Encoded as literal `'?'` strings with leading spaces.
   - Restricted strictly to categorical features: `occupation` (5.66%), `workclass` (5.64%), and `native_country` (1.79%).
3. **Census Privacy Top-Coding:**
   - `capital_gain` artificially capped at **\$99,999**, creating an artificial spike at the right boundary.
4. **Exact Feature Redundancy:**
   - `education` and `education_num` have a 1-to-1 rank correspondence (perfect multicollinearity).
5. **High Cardinality Imbalance:**
   - `native_country` had 42 distinct countries, with 89.6% belonging to `United-States`.

---

## Slide 5: Data Cleaning & Transformation (Methodological Justification)

| Challenge | Applied Technique | Methodological Justification |
| :--- | :--- | :--- |
| **Missing `'?'` Values** | Explicit Category `'Unknown'` | Preserves 100% of observations; avoids introducing artificial certainty via mode imputation in informative non-responses. |
| **Multicollinearity** | Drop `education`, retain `education_num` | Eliminates rank-redundant features while preserving ordinal academic progression. |
| **Noisy Survey Weight** | Drop `fnlwgt` | `fnlwgt` is a demographic sampling weight that adds geographical variance noise without intrinsic individual predictive signal. |
| **High Cardinality** | Regional Grouping (`United_States`, `Latin_America`, `Asia`, `Europe`, `Other`) | Reduces dimensionality, prevents matrix sparsity, and preserves geopolitical economic groupings. |
| **Right-Skewed Wealth** | `log1p` transformation + binary indicators (`has_capital_gain`) | Compresses heavy right tails and stabilizes variance across orders of magnitude. |
| **Feature Scaling** | `RobustScaler` (Median & IQR) | Resistant to extreme outliers compared to `StandardScaler` (Z-score). |

---

## Slide 6: Domain Feature Engineering
- **`net_capital`**: $capital\_gain - capital\_loss$ (net annual financial asset return).
- **`is_overtime`**: Binary indicator ($hours\_per\_week > 40$).
- **`hours_category`**: Part-time (<35h), Full-time (35–40h), Overtime (41–50h), Extreme (>50h).
- **`age_group`**: Young (<25), Prime Career (25–45), Mature Career (46–65), Senior (>65).
- **`marital_group`**: Consolidated marital statuses into Married, Never Married, Separated/Divorced, Widowed.

---

## Slide 7: Unsupervised Segmentation & Bonus Models Comparison
- **Model 1: K-Means Clustering** ($k=3$, evaluated via Elbow Method and Silhouette = **0.936**).
- **Model 2 (BONUS ⭐): Hierarchical Agglomerative Clustering (CAH)**:
  - Ward linkage with **Dendrogram visualization** (dissimilarity threshold cut-off at distance = 150).
  - Silhouette = **0.935**, validating the deterministic hierarchy of the clusters.
- **Model 3 (BONUS ⭐): DBSCAN (Density-Based Clustering)**:
  - Detected 2 dense core clusters and isolated **15.7% of points as density outliers/noise**, reinforcing our outlier diagnostic findings.
- **Emerging Socioeconomic Personas:**
  - **Persona 1 (Young Hourly Workers):** Avg age 27, 9.2 years education, 38h/wk, <10% earning >50K.
  - **Persona 2 (Prime White-Collar Professionals):** Avg age 44, 13+ years education, 45h/wk, >52% earning >50K.
  - **Persona 3 (Mid-Career Blue-Collar / Service):** Avg age 41, 10 years education, 40h/wk, ~25% earning >50K.

---

## Slide 8: Supervised Classification Benchmarking

Evaluated on untouched test split (16,281 observations):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 81.2% | 58.4% | 82.1% | 0.682 | 0.9097 |
| **Decision Tree (depth=8)** | 82.0% | 61.2% | 72.4% | 0.664 | 0.8926 |
| **Random Forest (depth=12)** | 84.1% | 63.8% | 75.9% | 0.693 | 0.9192 |
| **HistGradientBoosting (Champion)** | **87.2%** | **78.1%** | **64.8%** | **0.708** | **0.9262** |

- **Champion Model:** `HistGradientBoosting` achieved the highest discriminative power with an **ROC-AUC of 0.9262** and overall accuracy of **87.2%**.

---

## Slide 9: Feature Importance & Interpretability
1. **Marital Status (`marital_group_Married`):** Top driver of high earning status, reflecting dual-earner financial consolidation.
2. **Education Level (`education_num`):** Strongest monotonic relationship with income progression.
3. **Financial Capital (`net_capital` & `log_capital_gain`):** Powerful deterministic separator for the top income bracket.
4. **Labor Input (`hours_per_week` & `is_overtime`):** Essential baseline requirement for high earnings.

---

## Slide 10: Conclusion & Business Recommendations
- **Summary:** Clean data preparation, domain feature engineering, and robust scaling boosted baseline performance into production-ready territory (ROC-AUC 0.926+).
- **Practical Deployment:**
  - Use `HistGradientBoosting` for high-precision credit applicant triage.
  - Tune classification probability thresholds depending on business risk tolerance (lowering threshold to 0.35 if minimizing default risk is paramount).
- **Deliverables Completed:**
  - Fully documented Jupyter Notebook (`01_adult_census_data_prep_modeling.ipynb`) with bonus clustering models.
  - Cleaned train and test CSV files (`adult_cleaned_train.csv`, `adult_cleaned_test.csv`).
  - Complete 10-Minute Slide Deck for oral defense.
