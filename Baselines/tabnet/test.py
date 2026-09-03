import pandas as pd
import joblib
import sys
import os
import data_preprocessing as dp
import ensemble_classifier as ec


EXTERNAL_DATASET_PATH = '/data/verified_Phishtank.csv' 
MODEL_PATH = 'PhishFusion_phishing_detection_model.pkl'

def main():
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Run train.py first.")
        sys.exit(1)
        
    if not os.path.exists(EXTERNAL_DATASET_PATH):
        print(f"Error: Dataset file '{EXTERNAL_DATASET_PATH}' not found.")
        sys.exit(1)

    print(f"Loading Model from {MODEL_PATH}...")
    artifacts = joblib.load(MODEL_PATH)
    
    model = artifacts['model']
    scaler = artifacts['scaler']
    imputer = artifacts['imputer']
    selected_features = artifacts['selected_features']
    
    print(f"Loading External Data from {EXTERNAL_DATASET_PATH}...")
    df_test = pd.read_csv(EXTERNAL_DATASET_PATH)
    
    if 'url' not in df_test.columns or 'phishing' not in df_test.columns:
        raise ValueError("External dataset must contain 'url' and 'phishing' columns.")
    
 
    X_test_final = dp.process_new_data(df_test, imputer, scaler, selected_features)
    y_test_true = df_test['phishing'].values
    

    
    print("\nRunning Predictions...")
    y_pred = model.predict(X_test_final)
    y_prob = model.predict_proba(X_test_final)     # <--- Add this line
    
    
    ec.evaluate_metrics(y_test_true, y_pred, y_prob=y_prob, label="External Dataset")
    
    
    df_test['predicted_phishing'] = y_pred
    df_test.to_csv('external_predictions_result.csv', index=False)
    print("\nPredictions saved to 'external_predictions_result.csv'")

if __name__ == "__main__":
    main()
    
