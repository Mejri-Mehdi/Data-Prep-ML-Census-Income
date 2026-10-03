# 📊 Adult Census Income: Data Preparation, Clustering & Classification

**Institution:** ESPRIT — 4CCE9 (2026–2027)  
**Lab:** Data Preparation & Predictive Modeling  
**Instructor:** Mohamed Aziz KASSEB  
**Author(s):** [Your Name / Group Members]  
**Submission Deadline:** October 4, 2026  

---

## 📌 1. Project Overview & Business Understanding

Predicting individual income levels is a foundational problem in socioeconomic policy, financial risk assessment, and targeted marketing. Derived from the 1994 U.S. Census Bureau database by Ronny Kohavi and Barry Becker, the **Adult Census Income dataset** poses a binary classification challenge: 
> *Predict whether an individual's annual income exceeds $50,000 based on demographic, educational, and employment indicators.*

### 🎯 Business & Practical Objectives:
1. **Financial Inclusion & Credit Scoring:** Pre-qualifying individuals for financial products without requiring invasive tax audits.
2. **Socioeconomic Policy Formulation:** Understanding the primary drivers of wage disparities (education level, working hours, gender, and capital gains).
3. **Marketing Segmentation:** Targeting premium services to high-earning consumer segments.

---

## 🏗️ 2. Project Architecture & Directory Structure

```text
data-prep-census-income/
│
├── data/
│   ├── raw/                                     # Raw, untouched UCI files (adult.data, adult.test)
│   └── processed/                               # Cleaned, engineered, and scaled datasets (.csv)
│
├── notebooks/
│   └── 01_adult_census_data_prep_modeling.ipynb # Comprehensive end-to-end CRISP-DM notebook
│
├── presentations/
│   └── presentation_slides.md                   # 10-minute executive presentation deck
│
├── src/
│   ├── __init__.py
│   ├── data_cleaner.py                          # Imputation, outlier removal, transformations
│   ├── feature_engineer.py                      # Socioeconomic indicators, binning, encoding
│   └── evaluate.py                              # Metric calculations, confusion matrices, ROC/PR
│
├── requirements.txt                             # Pinned dependencies
└── README.md                                    # Project documentation
```

---

## 🔄 3. CRISP-DM Workflow Stages

This project strictly follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology:

1. **Business Understanding:** Problem framing, defining evaluation metrics (ROC-AUC, F1-Score for imbalanced classes).
2. **Data Understanding:** Inspecting schema (14 features + target), handling non-standard null values (`?`), detecting distribution skewness.
3. **Data Preparation (Core Focus):**
   - Imputation strategies justified by distribution nature (Categorical mode vs. conditional imputation).
   - Addressing severe skewness in `capital-gain` and `capital-loss` (90%+ zeros).
   - Resolving high cardinality in categorical features (`native-country`, `occupation`).
   - Normalization/Standardization via `RobustScaler` to protect against extreme outliers.
   - Engineering novel domain features (`net_capital`, `hours_category`, `education_tier`).
4. **Unsupervised Segmentation (Clustering):**
   - K-Means & PCA projection to uncover demographic personas.
   - Determining optimal $k$ using the Elbow Method and Silhouette Analysis.
5. **Supervised Classification:**
   - Benchmarking Logistic Regression, Decision Tree, and Random Forest / Gradient Boosting.
   - Stratified train/test splitting to preserve the 75/25 target ratio.
6. **Evaluation & Interpretation:**
   - Detailed metric reporting (Precision, Recall, F1, ROC-AUC).
   - Feature importance and business insights.

---

## ⚙️ 4. Quickstart Installation

```bash
# Clone the repository
git clone https://github.com/your-username/data-prep-census-income.git
cd data-prep-census-income

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```
