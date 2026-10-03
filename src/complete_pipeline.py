import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, classification_report, roc_auc_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def run():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data', 'raw', 'adult.data')
    test_path = os.path.join(base_dir, 'data', 'raw', 'adult.test')
    
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]
    
    df_train = pd.read_csv(train_path, header=None, names=columns, skipinitialspace=True)
    df_test = pd.read_csv(test_path, header=None, names=columns, skipinitialspace=True, skiprows=1)
    
    # 1. Strip trailing dot and encode target
    df_train['income'] = df_train['income'].str.rstrip('.')
    df_test['income'] = df_test['income'].str.rstrip('.')
    
    y_train = (df_train['income'] == '>50K').astype(int)
    y_test = (df_test['income'] == '>50K').astype(int)
    
    print(f"Target distribution (Train): {y_train.mean():.4f}")
    print(f"Target distribution (Test):  {y_test.mean():.4f}")
    
    # 2. Data Cleaning & Feature Engineering Function
    def preprocess_df(df):
        df = df.copy()
        
        # Replace '?' with 'Unknown'
        df['workclass'] = df['workclass'].replace('?', 'Unknown')
        df['occupation'] = df['occupation'].replace('?', 'Unknown')
        df['native_country'] = df['native_country'].replace('?', 'Unknown')
        
        # Group marital_status into simplified categories
        marital_map = {
            'Married-civ-spouse': 'Married',
            'Married-AF-spouse': 'Married',
            'Married-spouse-absent': 'Separated_Divorced',
            'Divorced': 'Separated_Divorced',
            'Separated': 'Separated_Divorced',
            'Widowed': 'Widowed',
            'Never-married': 'Never_Married'
        }
        df['marital_group'] = df['marital_status'].map(marital_map).fillna('Other')
        
        # Group native_country into regions
        def map_country(country):
            if country == 'United-States':
                return 'United_States'
            elif country in ['Mexico', 'Puerto-Rico', 'Cuba', 'Jamaica', 'Dominican-Republic', 
                             'Guatemala', 'El-Salvador', 'Columbia', 'Haiti', 'Nicaragua', 
                             'Peru', 'Ecuador', 'Trinadad&Tobago', 'Honduras']:
                return 'Latin_America_Caribbean'
            elif country in ['Philippines', 'India', 'China', 'Japan', 'Vietnam', 
                             'Taiwan', 'Iran', 'Hong', 'Thailand', 'Cambodia', 'Laos']:
                return 'Asia'
            elif country in ['Germany', 'England', 'Italy', 'Poland', 'Portugal', 
                             'Greece', 'Ireland', 'France', 'Yugoslavia', 'Scotland', 
                             'Hungary', 'Holand-Netherlands']:
                return 'Europe'
            elif country in ['Canada', 'Outlying-US(Guam-USVI-etc)', 'South']:
                return 'Other_NorthAmerica_Etc'
            else:
                return 'Unknown_Other'
                
        df['region'] = df['native_country'].apply(map_country)
        
        # Group workclass
        def map_workclass(wc):
            if wc in ['Federal-gov', 'Local-gov', 'State-gov']:
                return 'Government'
            elif wc in ['Self-emp-inc', 'Self-emp-not-inc']:
                return 'Self_Employed'
            elif wc == 'Private':
                return 'Private'
            else:
                return 'Other_Unknown'
        df['workclass_group'] = df['workclass'].apply(map_workclass)
        
        # Feature Engineering:
        df['net_capital'] = df['capital_gain'] - df['capital_loss']
        df['has_capital_gain'] = (df['capital_gain'] > 0).astype(int)
        df['has_capital_loss'] = (df['capital_loss'] > 0).astype(int)
        df['log_capital_gain'] = np.log1p(df['capital_gain'])
        df['log_capital_loss'] = np.log1p(df['capital_loss'])
        
        df['is_overtime'] = (df['hours_per_week'] > 40).astype(int)
        df['hours_category'] = pd.cut(
            df['hours_per_week'], 
            bins=[0, 34, 40, 50, 100], 
            labels=['Part_Time', 'Full_Time', 'Overtime', 'Extreme_Hours']
        ).astype(str)
        
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[16, 25, 45, 65, 100], 
            labels=['Young', 'Prime_Career', 'Mature_Career', 'Senior']
        ).astype(str)
        
        # Drop redundant / uninformative columns
        # Drop 'education' (redundant with education_num)
        # Drop 'fnlwgt' (survey weight, not individual socioeconomic signal)
        # Drop raw high-cardinality columns
        drop_cols = ['education', 'fnlwgt', 'marital_status', 'native_country', 'workclass', 'income']
        df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
        return df_clean

    X_train_raw = preprocess_df(df_train)
    X_test_raw = preprocess_df(df_test)
    
    # One-hot encoding on categorical columns
    cat_cols = X_train_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical columns to encode: {cat_cols}")
    
    # Use pandas get_dummies with alignment
    X_train_encoded = pd.get_dummies(X_train_raw, columns=cat_cols, drop_first=True, dtype=int)
    X_test_encoded = pd.get_dummies(X_test_raw, columns=cat_cols, drop_first=True, dtype=int)
    
    # Align test columns with train
    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)
    
    print(f"Encoded Train shape: {X_train_encoded.shape}")
    print(f"Encoded Test shape:  {X_test_encoded.shape}")
    
    # Scale numerical features with RobustScaler
    num_cols = ['age', 'education_num', 'hours_per_week', 'capital_gain', 'capital_loss', 
                'net_capital', 'log_capital_gain', 'log_capital_loss']
    
    scaler = RobustScaler()
    X_train_scaled = X_train_encoded.copy()
    X_test_scaled = X_test_encoded.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train_encoded[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test_encoded[num_cols])
    
    # Export clean datasets
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    train_out = os.path.join(processed_dir, 'adult_cleaned_train.csv')
    test_out = os.path.join(processed_dir, 'adult_cleaned_test.csv')
    
    # Attach target for export
    train_export = X_train_scaled.copy()
    train_export['income_target'] = y_train
    test_export = X_test_scaled.copy()
    test_export['income_target'] = y_test
    
    train_export.to_csv(train_out, index=False)
    test_export.to_csv(test_out, index=False)
    print(f"Cleaned datasets exported to {train_out} and {test_out}")
    
    # 3. K-Means Clustering on a sample of Train
    print("\n--- Running Unsupervised K-Means ---")
    sample_indices = np.random.RandomState(42).choice(len(X_train_scaled), size=5000, replace=False)
    X_sample = X_train_scaled.iloc[sample_indices]
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sample)
    print(f"PCA 2D explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_sample)
    sil_score = silhouette_score(X_sample, clusters)
    print(f"Silhouette Score (k=3): {sil_score:.4f}")
    
    # 4. Supervised Classification Benchmarking
    print("\n--- Benchmarking Supervised Models ---")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42)
    }
    
    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else preds
        
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        
        print(f"[{name}] ROC-AUC: {auc:.4f} | F1-Score: {f1:.4f}")
        results.append({'Model': name, 'ROC_AUC': round(auc, 4), 'F1_Score': round(f1, 4)})
        
    df_results = pd.DataFrame(results)
    print("\nModel Benchmarking Results:")
    print(df_results.to_string(index=False))

if __name__ == '__main__':
    run()
