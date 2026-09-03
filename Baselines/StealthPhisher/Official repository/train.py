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

# Ensure you have stealth_features.py in the same directory
from stealth_features import extract_features_parallel

# Configuration
DATASET_FILE = "/home/hussain-hu/phishing/github/data/StealthPhisher2025.csv"
CACHE_PATH = "extracted_features.csv"
MODEL_PATH = "ebbu2017_model.h5"
SCALER_PATH = "ebbu2017_scaler.pkl"
FEATURES_PATH = "ebbu2017_selected_features.json"

def log(msg):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculates metrics matching the paper's methodology and your requested format."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0

    # Extended Metrics (from StealthPhisher_Model_Selection.ipynb)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Markedness: (Precision * Recall) - ((1-Precision) * FNR)
    # Note: Using the explicit formula found in the paper's code snippets
    if (tp + fn) > 0 and (fn + tp) > 0:
        markedness = (prec * (tp / (tp + fn))) - ((1 - prec) * (fn / (fn + tp)))
    else:
        markedness = 0.0

    # Youden's J
    youdens_j = rec + specificity - 1
    
    # FMI
    fmi = np.sqrt(prec * rec)

    print("\n" + "="*30)
    print("FINAL MODEL PERFORMANCE")
    print("="*30)
    # Requested Metrics
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print("-" * 30)
    # Additional Paper Metrics
    print(f"MCC:         {mcc:.4f}")
    print(f"Markedness:  {markedness:.4f}")
    print(f"Youden's J:  {youdens_j:.4f}")
    print(f"FMI:         {fmi:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print("="*30 + "\n")

def perform_cspca(X, y, feature_names, threshold=0.015):
    log("Running Class-Specific PCA (CSPCA)...")
    scaler = StandardScaler()
    
    def get_variance(data, label_name):
        if len(data) == 0: return pd.Series(0, index=feature_names)
        X_scaled = scaler.fit_transform(data)
        pca = PCA(n_components=len(feature_names))
        pca.fit(X_scaled)
        return pd.Series(pca.explained_variance_ratio_, index=feature_names)

    X_benign = X[y == 0]
    X_phishing = X[y == 1]
    
    imp_benign = get_variance(X_benign, "Benign")
    imp_phishing = get_variance(X_phishing, "Phishing")
    
    avg_imp = (imp_benign + imp_phishing) / 2
    
    selected = avg_imp[avg_imp > threshold].index.tolist()
    log(f"CSPCA Selected {len(selected)}/{len(feature_names)} features")
    return selected

def build_model(input_dim):
    # Hybrid Wide & Deep Architecture
    wide_input = Input(shape=(input_dim,), name='wide_input')
    wide_output = Dense(1, activation='sigmoid')(wide_input)

    deep_input = Input(shape=(input_dim,), name='deep_input')
    deep_layer_1 = Dense(64, activation='relu')(deep_input)
    deep_layer_2 = Dense(32, activation='relu')(deep_layer_1)
    deep_output = Dense(1, activation='sigmoid')(deep_layer_2)

    concatenated = Concatenate()([wide_output, deep_output])
    final_output = Dense(1, activation='sigmoid')(concatenated)

    model = Model(inputs=[wide_input, deep_input], outputs=final_output)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def main():
    log("--- StealthPhisher Training Pipeline ---")

    # 1. Load Data
    if os.path.exists(CACHE_PATH):
        log(f"Loading cached features from {CACHE_PATH}...")
        df = pd.read_csv(CACHE_PATH)
    else:
        log(f"Loading raw dataset: {DATASET_FILE}...")
        try:
            raw_df = pd.read_csv(DATASET_FILE)
            # Normalize column names to handle variations
            raw_df.columns = [c.lower() for c in raw_df.columns]
            
            # Identify URL and Label columns
            url_col = next((c for c in raw_df.columns if 'url' in c), None)
            label_col = next((c for c in raw_df.columns if 'phish' in c or 'label' in c or 'class' in c), None)
            
            if not url_col or not label_col:
                log("Error: Could not automatically find 'url' or 'label' columns in CSV.")
                return

            raw_df.rename(columns={url_col: 'url', label_col: 'phishing'}, inplace=True)
            
            # Run extraction
            log("Starting feature extraction (50 workers)...")
            df = extract_features_parallel(raw_df, url_col='url', workers=50)
            df.to_csv(CACHE_PATH, index=False)
            
        except Exception as e:
            log(f"Error loading dataset: {e}")
            return

    df.dropna(inplace=True)
    
    # 2. Preparation
    from stealth_features import StealthPhisherExtractor
    extractor = StealthPhisherExtractor()
    valid_features = [c for c in extractor.feature_columns if c in df.columns]

    X = df[valid_features]
    y = df['phishing']
    
    # 3. Splitting (Hold-out Validation)
    log("Splitting: 80% Train, 20% Hold-out Test")
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    log("Splitting Train: 80% Train, 20% Val (for Early Stopping)")
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)
    
    # 4. Feature Selection
    selected_features = perform_cspca(X_train, y_train, valid_features)
    with open(FEATURES_PATH, 'w') as f:
        json.dump(selected_features, f)

    # 5. Scaling
    log("Fitting Scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[selected_features])
    X_val_scaled = scaler.transform(X_val[selected_features])
    X_test_scaled = scaler.transform(X_test[selected_features])
    
    joblib.dump(scaler, SCALER_PATH)

    # 6. Training
    log("Building Model...")
    model = build_model(len(selected_features))
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1)
    ]
    
    log("Starting Training...")
    model.fit(
        [X_train_scaled, X_train_scaled], y_train,
        validation_data=([X_val_scaled, X_val_scaled], y_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. Evaluation
    log("Evaluating on Hold-out Test Set...")
    # Get probabilities for AUC
    y_pred_prob = model.predict([X_test_scaled, X_test_scaled])
    # Get binary predictions for Accuracy/F1
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    calculate_metrics(y_test, y_pred, y_pred_prob)
    log(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()