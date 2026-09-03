import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix
)
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import features  

DATASET_PATH = '/data/ebbu2017.csv'  # Replace with your dataset file
MODEL_PATH = 'ebbu2017_lgbm_model.pkl'

def train():
    print(f"1. Loading dataset from {DATASET_PATH}...")
    try:
        # Load data: Expects columns 'url' and 'phishing'
        df = pd.read_csv(DATASET_PATH, encoding='ISO-8859-1')
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return

    print(f"   Shape: {df.shape}")
    
    print("2. Extracting features (this may take time)...")
    X = features.extract_features(df)
    y = df['phishing']

    print("3. Splitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("4. Applying SMOTE (Synthetic Minority Over-sampling)...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    print(f"   Original training samples: {len(y_train)}")
    print(f"   Resampled training samples: {len(y_train_res)}")

    print("5. Training LightGBM Model...")
    
    lgb = LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        n_jobs=5,
        random_state=5
    )
    lgb.fit(X_train_res, y_train_res)

    print("6. Evaluating Model...")
    
    y_pred = lgb.predict(X_test)
    
    y_prob = lgb.predict_proba(X_test)[:, 1]
    
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc_auc = 0.0  

    
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC AUC:   {roc_auc:.4f}")

    print("\n   Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred))

    print(f"7. Saving model to {MODEL_PATH}...")
    joblib.dump(lgb, MODEL_PATH)
    print("   Done.")

if __name__ == "__main__":
    train()