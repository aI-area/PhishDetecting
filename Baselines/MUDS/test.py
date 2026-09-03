import pandas as pd
import joblib
import os
import features 
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

MODEL_PATH = 'ebbu2017_lgbm_model.pkl'

TEST_DATA_PATH = '/data/phishstorm.csv' 
OUTPUT_PATH = 'predictions.csv'

def test():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Run train.py first.")
        return

    print(f"1. Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    print(f"2. Loading test data from {TEST_DATA_PATH}...")
    if not os.path.exists(TEST_DATA_PATH):
        print("Error: Test data file not found.")
        return
        
    df = pd.read_csv(TEST_DATA_PATH)
    if 'url' not in df.columns:
        print("Error: Dataset must have a 'url' column.")
        return

    print("3. Extracting features...")
    try:
        X_new = features.extract_features(df)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return

    print("4. Predicting...")
    
    predictions = model.predict(X_new)
    
 
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_new)[:, 1]
    
 
    df['predicted_label'] = predictions
    df['prediction_class'] = df['predicted_label'].map({0: 'Legitimate', 1: 'Phishing'})
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"5. Results saved to {OUTPUT_PATH}")
    
   
    if 'phishing' in df.columns:
        print("\n--- Evaluation on Test Data ---")
        y_true = df['phishing']
        
        acc = accuracy_score(y_true, predictions)
        prec = precision_score(y_true, predictions, average='weighted', zero_division=0)
        rec = recall_score(y_true, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
        
        
        roc_val = 0.0
        if probs is not None:
            try:
                roc_val = roc_auc_score(y_true, probs)
            except ValueError:
               
                print("   ROC AUC:   N/A (Only one class present in test data)")
                roc_val = 0.0

        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        if probs is not None and roc_val > 0:
            print(f"   ROC AUC:   {roc_val:.4f}")
        
        print("\n   Confusion Matrix:")
        print(confusion_matrix(y_true, predictions))
        
        print("\n   Classification Report:")
        print(classification_report(y_true, predictions))
    else:
        print("\nNote: Ground truth column 'phishing' not found in test data. Skipping metrics.")

if __name__ == "__main__":
    test()