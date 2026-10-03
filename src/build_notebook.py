import os
import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: Markdown - Title & Business Understanding
c1 = nbf.v4.new_markdown_cell("""# 📊 Lab: Data Preparation, Clustering & Classification
**Course:** ESPRIT — 4CCE9 (2026–2027)  
**Dataset:** Adult Census Income (UCI Machine Learning Repository)  
**Methodology:** CRISP-DM (Cross-Industry Standard Process for Data Mining)  
**Target Variable:** `income` (`<=50K` vs `>50K`)

---

## 1. Business Understanding

### 1.1 Context & Problem Formulation
Predicting an individual's earning capacity based on demographic, occupational, and educational parameters is a core problem in computational economics, financial credit scoring, and government policy planning. The **Adult Census Income dataset** (extracted from the 1994 U.S. Census Bureau database by Ronny Kohavi & Barry Becker) frames this as a binary classification challenge:
> **Objective:** Predict whether an individual's annual income exceeds $50,000.

### 1.2 Practical Business Applications
1. **Credit Risk & Loan Pre-Approval:** Allows automated financial underwriting without requiring immediate tax audit disclosures.
2. **Public Policy & Wage Disparity Research:** Dissects the quantitative relationship between education, working hours, demographic factors, and income thresholds.
3. **Targeted Wealth Marketing:** Enables institutions to direct wealth management and premium services to high-earning brackets.

### 1.3 Asymmetric Error Costs & Metric Justification
- **False Positive (Type I Error):** Model predicts `>50K`, but the true income is `<=50K`. (Consequence: Granting a loan or credit limit beyond the applicant's repayment ability $\\rightarrow$ high default risk).
- **False Negative (Type II Error):** Model predicts `<=50K`, but the true income is `>50K`. (Consequence: Sub-optimal marketing targeting / minor lost opportunity).
- **Metric Choice:** Because the dataset has ~76% `<=50K` and ~24% `>50K`, standard **Accuracy is misleading** (a naive model predicting `<=50K` for everyone achieves ~76% accuracy while catching 0% of high earners). We prioritize **ROC-AUC, Precision-Recall AUC (PR-AUC), and F1-Score**.
""")

# Cell 2: Code - Imports & Setup
c2 = nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11

print("Libraries successfully imported!")
""")

# Cell 3: Markdown - Data Loading
c3 = nbf.v4.new_markdown_cell("""---
## 2. Data Understanding & Acquisition

We load both the training split (`adult.data`, 32,561 rows) and test split (`adult.test`, 16,281 rows).
**Key Considerations:**
- Column headers are missing in the raw text and must be assigned from `adult.names`.
- Leading whitespace after commas must be stripped (`skipinitialspace=True`).
- Line 1 of `adult.test` contains a cross-validation comment (`|1x3 Cross validator`), requiring `skiprows=1`.
""")

# Cell 4: Code - Data Loading
c4 = nbf.v4.new_code_cell("""columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

train_path = '../data/raw/adult.data'
test_path  = '../data/raw/adult.test'

# Load raw splits
df_train = pd.read_csv(train_path, header=None, names=columns, skipinitialspace=True)
df_test  = pd.read_csv(test_path,  header=None, names=columns, skipinitialspace=True, skiprows=1)

print(f"Training set shape: {df_train.shape}")
print(f"Test set shape:     {df_test.shape}")
print(f"Total observations: {len(df_train) + len(df_test)}")

df_train.head()
""")

# Cell 5: Markdown - Data Traps Diagnosis
c5 = nbf.v4.new_markdown_cell("""---
### 2.1 Unmasking Data Quality Issues & Traps

We immediately inspect:
1. **Target String Inconsistency:** Note that UCI added a trailing dot `.` to labels in `adult.test` (`<=50K.` vs `<=50K`).
2. **Hidden Missing Values (`?`):** Missing entries were encoded as literal `'?'` characters, escaping standard `.isnull()` checks.
3. **Exact Duplicates:** Identifying identical participant responses.
""")

# Cell 6: Code - Quality Inspection
c6 = nbf.v4.new_code_cell("""# 1. Target Label Comparison
print("=== TARGET VALUE COUNTS (Train) ===")
print(df_train['income'].value_counts())
print("\\n=== TARGET VALUE COUNTS (Test) ===")
print(df_test['income'].value_counts())

# 2. Inspecting Hidden Missing Values ('?')
print("\\n=== FEATURES WITH '?' ENCODED VALUES (Train) ===")
missing_stats = []
for col in df_train.columns:
    q_count = (df_train[col] == '?').sum()
    if q_count > 0:
        missing_stats.append({
            'Feature': col,
            'Missing_Count': q_count,
            'Percentage': round((q_count / len(df_train)) * 100, 2)
        })

df_missing = pd.DataFrame(missing_stats)
display(df_missing)

# 3. Duplicate Records
print(f"\\nDuplicate rows in Train: {df_train.duplicated().sum()}")
print(f"Duplicate rows in Test:  {df_test.duplicated().sum()}")
""")

# Cell 7: Markdown - Exploratory Visualizations
c7 = nbf.v4.new_markdown_cell("""---
## 3. Exploratory Data Analysis (EDA) & Visualizations

We now systematically analyze:
- **Target Distribution & Imbalance**
- **Age Distribution by Income Level**
- **Education Level & Educational Attainment (`education_num`) vs Income**
- **Working Hours (`hours_per_week`) vs Income**
- **Extreme Skewness in `capital_gain` and `capital_loss`**
""")

# Cell 8: Code - Visualizations 1 & 2
c8 = nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Target Class Imbalance
sns.countplot(data=df_train, x='income', ax=axes[0], palette=['#4A90E2', '#E74C3C'])
axes[0].set_title('Target Class Distribution (Train Set)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Income Class')
axes[0].set_ylabel('Number of Individuals')
total = len(df_train)
for p in axes[0].patches:
    pct = f"{100 * p.get_height() / total:.1f}%"
    axes[0].annotate(f"{p.get_height():,} ({pct})", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', color='white', fontweight='bold')

# Plot 2: Age vs Income KDE
sns.kdeplot(data=df_train, x='age', hue='income', common_norm=False, fill=True, ax=axes[1], palette=['#4A90E2', '#E74C3C'], alpha=0.4)
axes[1].set_title('Age Distribution by Income Level', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Density')

plt.tight_layout()
plt.show()
""")

# Cell 9: Code - Visualizations 3 & 4
c9 = nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot 3: Education Years vs Income Proportion
edu_income = pd.crosstab(df_train['education_num'], df_train['income'], normalize='index')
edu_income.plot(kind='bar', stacked=True, ax=axes[0], color=['#4A90E2', '#E74C3C'])
axes[0].set_title('Proportion of >50K by Years of Education (education_num)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Years of Education (education_num)')
axes[0].set_ylabel('Proportion')
axes[0].legend(title='Income')
axes[0].tick_params(axis='x', rotation=0)

# Plot 4: Hours Per Week vs Income
sns.boxplot(data=df_train, x='income', y='hours_per_week', ax=axes[1], palette=['#4A90E2', '#E74C3C'])
axes[1].set_title('Working Hours Distribution by Income Level', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Income')
axes[1].set_ylabel('Hours Worked Per Week')

plt.tight_layout()
plt.show()
""")

# Cell 10: Code - Skewness in Capital Gains & Losses
c10 = nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Capital Gain Distribution
sns.histplot(df_train['capital_gain'], bins=40, kde=False, ax=axes[0], color='#2ECC71')
axes[0].set_title('Capital Gain Distribution (Extreme 0-Inflation & Top-Coding)', fontsize=12, fontweight='bold')
axes[0].set_yscale('log')
axes[0].set_ylabel('Log Count')

# Capital Loss Distribution
sns.histplot(df_train['capital_loss'], bins=40, kde=False, ax=axes[1], color='#E67E22')
axes[1].set_title('Capital Loss Distribution (Extreme 0-Inflation)', fontsize=12, fontweight='bold')
axes[1].set_yscale('log')
axes[1].set_ylabel('Log Count')

plt.tight_layout()
plt.show()

print(f"Capital Gain Skewness: {df_train['capital_gain'].skew():.2f}")
print(f"Capital Loss Skewness: {df_train['capital_loss'].skew():.2f}")
print(f"Percentage of 0s in capital_gain: {(df_train['capital_gain'] == 0).mean() * 100:.2f}%")
print(f"Percentage of 0s in capital_loss: {(df_train['capital_loss'] == 0).mean() * 100:.2f}%")
""")

# Cell 11: Markdown - Summary of Findings & Next Steps
c11 = nbf.v4.new_markdown_cell("""---
### 3.1 Key Observations from EDA (For Report & Slides)
1. **Class Imbalance:** 75.9% of workers earn $\le$50K, while only 24.1% earn >50K. 
2. **Age Dynamics:** Higher earners have a peak distribution between ages 35 and 52, whereas lower earners are heavily concentrated among younger workers (<30).
3. **Education Threshold Effect:** High-earning probability accelerates sharply after 12 years of education (`education_num` $\ge$ 13 corresponds to Bachelor's, Master's, Doctorate).
4. **Hours Per Week Outliers:** The standard work week is 40 hours, but significant outliers work up to 99 hours or as low as 1 hour.
5. **Zero-Inflation in Capital Gains/Losses:** Over 91.7% of entries have 0 capital gains, and 95.3% have 0 capital losses. Furthermore, `capital_gain` is top-coded at $99,999.
6. **Redundancy:** `education` and `education_num` represent the identical underlying metric, requiring removal of one to prevent multicollinearity.
""")

nb.cells = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11]

target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
notebook_path = os.path.join(target_dir, 'notebooks', '01_adult_census_data_prep_modeling.ipynb')

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully generated: {notebook_path}")
