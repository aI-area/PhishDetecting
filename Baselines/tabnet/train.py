import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import data_preprocessing as dp
import feature_selection as fs
import ensemble_classifier as ec


DATASET_PATH = '/data/PhishFusion.csv'       
MODEL_OUTPUT_PATH = 'PhishFusion_phishing_detection_model.pkl'

def main():
    
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    
    
    if 'url' not in df.columns or 'phishing' not in df.columns:
        raise ValueError("Dataset must contain 'url' and 'phishing' columns.")

    X_raw = df 
    y = df['phishing'].values

    
    X_scaled_full, imputer, scaler, all_feature_names = dp.get_processing_artifacts(df)

   
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_full, y, test_size=0.2, random_state=42, stratify=y
    )

  
    top_indices, selected_features = fs.run_tabnet_feature_selection(X_train, y_train, all_feature_names)
    
 
    X_train_selected = X_train[:, top_indices]
    X_test_selected = X_test[:, top_indices]

 
    model = ec.train_model(X_train_selected, y_train)

 
    y_pred = model.predict(X_test_selected)
    y_prob = model.predict_proba(X_test_selected)
    ec.evaluate_metrics(y_test, y_pred, y_prob=y_prob, label="Validation Set (20%)")
   

    if len(X_test_selected) > 0:
        ec.explain_with_lime(model, X_train_selected, selected_features, X_test_selected[0])

    artifacts = {
        'model': model,
        'scaler': scaler,
        'imputer': imputer,
        'selected_features': selected_features
    }
    joblib.dump(artifacts, MODEL_OUTPUT_PATH)
    print(f"\nModel and processing artifacts saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()