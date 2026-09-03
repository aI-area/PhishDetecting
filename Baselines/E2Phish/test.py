import pandas as pd
import joblib
import argparse
import feature_extractor
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score

def test_model(test_csv_path):
    
    print("Loading model...")
    try:
        model = joblib.load('ebbu2017_e2phish_model.pkl')
        selected_features = joblib.load('ebbu2017_selected_features.pkl')
    except FileNotFoundError:
        print("Error: Model files not found. Please run train.py first.")
        return

    # 2. Load New Data
    print(f"Loading test data from {test_csv_path}...")
    try:
        new_data = pd.read_csv(test_csv_path)
        if 'url' not in new_data.columns or 'phishing' not in new_data.columns:
            print("Error: Test dataset must contain 'url' and 'phishing' columns.")
            return
    except FileNotFoundError:
        print(f"Error: File not found at {test_csv_path}")
        return

    # 3. Extract Features
    print("Extracting features from new URLs...")
    processed_test = feature_extractor.extract_features_from_dataframe(new_data)
    
    if processed_test.empty:
        print("No valid features extracted from the test set.")
        return

    
    X_test = pd.DataFrame(index=processed_test.index)
    for feature in selected_features:
        if feature in processed_test.columns:
            X_test[feature] = processed_test[feature]
        else:
            X_test[feature] = 0 
            
    y_test = processed_test['labels']

   
    print("Predicting...")
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    
    print("\n--- Test Set Performance ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    
    try:
        roc = roc_auc_score(y_test, y_probs)
        print(f"ROC AUC:   {roc:.4f}")
    except ValueError:
        print("ROC AUC:   N/A (Test set must contain both classes)")

    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    
    import sys
    
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        
        path = '/data/phishstorm.csv' 
    
    
    test_model(path)