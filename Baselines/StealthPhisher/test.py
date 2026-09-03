import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import os
import time
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, matthews_corrcoef, roc_auc_score, 
                             confusion_matrix)
from stealth_features import extract_features_parallel

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Configuration
TEST_DATASET = "/data/verified_Phishtank.csv" 
MODEL_PATH = "PhishFusion_model.h5"
SCALER_PATH = "PhishFusion_scaler.pkl"
FEATURES_PATH = "PhishFusion_selected_features.json"
OUTPUT_FILE = "PhishFusion_test_results.csv"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def calculate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    try: auc = roc_auc_score(y_true, y_prob)
    except: auc = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    if (tp + fn) > 0 and (fn + tp) > 0:
        markedness = (prec * (tp / (tp + fn))) - ((1 - prec) * (fn / (fn + tp)))
    else: markedness = 0.0
    youdens_j = rec + specificity - 1
    fmi = np.sqrt(prec * rec)

    print("\n" + "="*30)
    print("TEST SET PERFORMANCE")
    print("="*30)
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"ROC AUC:     {auc:.4f}")
    print("-" * 30)
    print(f"MCC:         {mcc:.4f}")
    print(f"Markedness:  {markedness:.4f}")
    print(f"Youden's J:  {youdens_j:.4f}")
    print(f"FMI:         {fmi:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print("="*30 + "\n")

def main():
    log("--- StealthPhisher Inference ---")
    if not os.path.exists(MODEL_PATH):
        log("Error: Artifacts not found. Run train.py first.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH, 'r') as f:
        selected_features = json.load(f)

    try:
        raw_df = pd.read_csv(TEST_DATASET)
        raw_df.columns = [c.lower() for c in raw_df.columns]
        url_col = next((c for c in raw_df.columns if 'url' in c), None)
        if not url_col: return
        raw_df.rename(columns={url_col: 'url'}, inplace=True)
    except Exception as e:
        log(f"Data error: {e}")
        return

    df = extract_features_parallel(raw_df, 'url', 50)

    # Fill and Select Features
    for col in selected_features:
        if col not in df.columns: df[col] = 0
    X_test = df[selected_features].values
    X_test_scaled = scaler.transform(X_test)

    y_pred_prob = model.predict([X_test_scaled, X_test_scaled])
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    df['Prediction'] = y_pred
    df['Confidence'] = y_pred_prob
    df.to_csv(OUTPUT_FILE, index=False)

    label_col = next((c for c in df.columns if ('phish' in c or 'label' in c) and 'pred' not in c), None)
    if label_col:
        y_true = df[label_col].values
        if y_true.dtype == object:
             y_true = np.where(pd.Series(y_true).astype(str).str.lower().str.contains('phish|1|malicious'), 1, 0)
        else:
             y_true = y_true.astype(int)
        calculate_metrics(y_true, y_pred, y_pred_prob)

    log(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()