import pandas as pd
import joblib
import sys
import pdd_impl
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score

def test_model(test_csv_path):
    print("Loading model...")
    try:
        model = joblib.load('PhishFusion_model.pkl')
        feature_names = joblib.load('PhishFusion_feature_names.pkl')
    except FileNotFoundError:
        print("Model files not found. Run train.py first.")
        return

    print(f"Loading test data from {test_csv_path}...")
    try:
        new_data = pd.read_csv(test_csv_path, encoding='ISO-8859-1')
        if 'url' not in new_data.columns or 'phishing' not in new_data.columns:
            print("Error: Dataset must have 'url' and 'phishing' columns.")
            return
    except FileNotFoundError:
        print(f"File not found: {test_csv_path}")
        return

    # Extract Features
    processed_test = pdd_impl.extract_features_from_dataframe(new_data)
    
    if processed_test.empty:
        print("No valid features extracted.")
        return


    X_test = pd.DataFrame(index=processed_test.index)
    for feat in feature_names:
        X_test[feat] = processed_test[feat] if feat in processed_test.columns else 0
        
    y_test = processed_test['labels']

    print("Predicting...")
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    print("\n--- Test Results ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    try:
        print(f"ROC AUC:   {roc_auc_score(y_test, y_probs):.4f}")
    except:
        pass
    
 

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = '/data/verified_Phishtank.csv'
        
    test_model(path)