import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, matthews_corrcoef, roc_auc_score, 
                             confusion_matrix)
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from stealth_features import extract_features_parallel

# Configuration
DATASET_FILE = "/home/hussain-hu/phishing/github/data/PhishFusion.csv"
CACHE_PATH = "PhishFusion_extracted_features.csv" # The file causing the issue
MODEL_PATH = "PhishFusion_model.h5"
SCALER_PATH = "PhishFusion_scaler.pkl"
FEATURES_PATH = "PhishFusion_selected_features.json"

# --- CRITICAL FIX: FORCE FRESH EXTRACTION ---
# This ensures we don't accidentally load old/bad features
if os.path.exists(CACHE_PATH):
    print(f"[WARNING] Deleting old cache '{CACHE_PATH}' to force fresh extraction...")
    os.remove(CACHE_PATH)
# --------------------------------------------

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")

def perform_cspca(X, y, feature_names, threshold=0.015):
    log("Running CSPCA Feature Selection...")
    
    # 1. DROP CONSTANT COLUMNS (Crucial for Offline Mode)
    # Features like 'HasFavicon' are always 0 in offline mode, so we must drop them
    # otherwise StandardScaler divides by zero.
    valid_cols = [c for c in feature_names if X[c].std() > 0]
    log(f"Dropped {len(feature_names) - len(valid_cols)} constant features (Offline Mode safe).")
    
    scaler = StandardScaler()
    X_clean = X[valid_cols]
    
    def get_variance(data):
        if len(data) == 0: return pd.Series(0, index=valid_cols)
        X_scaled = scaler.fit_transform(data)
        pca = PCA(n_components=min(len(data), len(valid_cols)))
        pca.fit(X_scaled)
        return pd.Series(pca.explained_variance_ratio_[0], index=valid_cols)

    imp_benign = get_variance(X_clean[y==0])
    imp_phishing = get_variance(X_clean[y==1])
    avg_imp = (imp_benign + imp_phishing) / 2
    
    selected = avg_imp[avg_imp > threshold].index.tolist()
    if not selected: selected = valid_cols 
        
    log(f"CSPCA Selected {len(selected)} features")
    return selected

def build_model(input_dim):
    wide_input = Input(shape=(input_dim,), name='wide_input')
    wide_output = Dense(1, activation='sigmoid')(wide_input)
    deep_input = Input(shape=(input_dim,), name='deep_input')
    deep_layer_1 = Dense(64, activation='relu')(deep_input)
    deep_layer_2 = Dense(32, activation='relu')(deep_layer_1)
    deep_output = Dense(1, activation='sigmoid')(deep_layer_2)
    out = Dense(1, activation='sigmoid')(Concatenate()([wide_output, deep_output]))
    model = Model(inputs=[wide_input, deep_input], outputs=out)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

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
    print("FINAL MODEL PERFORMANCE (Validation)")
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
    log("--- Training Pipeline ---")
    
    # 1. Load Data & Extract Features
    # The cache deletion block at the top ensures this runs fresh
    try:
        raw_df = pd.read_csv(DATASET_FILE, encoding='ISO-8859-1')
        raw_df.columns = [c.lower() for c in raw_df.columns]
        url_col = next((c for c in raw_df.columns if 'url' in c), None)
        label_col = next((c for c in raw_df.columns if 'phish' in c or 'label' in c), None)
        
        if not url_col or not label_col:
            log("Error: 'url' or 'label' column not found.")
            return

        raw_df.rename(columns={url_col: 'url', label_col: 'phishing'}, inplace=True)
        log("Starting fresh feature extraction (Fast Mode)...")
        # Ensure your stealth_features.py is the offline version
        df = extract_features_parallel(raw_df, url_col='url', workers=50)
        df.to_csv(CACHE_PATH, index=False)
    except Exception as e:
        log(f"Error loading dataset: {e}")
        return

    df.dropna(inplace=True)
    from stealth_features import StealthPhisherExtractor
    all_feats = StealthPhisherExtractor().feature_columns
    valid_feats = [c for c in all_feats if c in df.columns]

    X = df[valid_feats]
    y = df['phishing']

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    # 3. Feature Selection
    selected_features = perform_cspca(X_train, y_train, valid_feats)
    with open(FEATURES_PATH, 'w') as f: json.dump(selected_features, f)

    # 4. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[selected_features])
    X_val_scaled = scaler.transform(X_val[selected_features])
    X_test_scaled = scaler.transform(X_test[selected_features])
    joblib.dump(scaler, SCALER_PATH)

    # 5. Train
    model = build_model(len(selected_features))
    cb = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=0)]
    
    model.fit([X_train_scaled, X_train_scaled], y_train, 
              validation_data=([X_val_scaled, X_val_scaled], y_val),
              epochs=50, batch_size=64, callbacks=cb, verbose=1)

    y_pred_prob = model.predict([X_test_scaled, X_test_scaled])
    calculate_metrics(y_test, (y_pred_prob > 0.5).astype(int), y_pred_prob)
    log(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()