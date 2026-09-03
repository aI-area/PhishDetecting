import pandas as pd
import joblib
import pdd_impl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score


dataset_path = '/data/PhishFusion.csv'
print(f"Loading Data from {dataset_path}...")

try:
    
    data = pd.read_csv(dataset_path, encoding='ISO-8859-1')
    if 'url' not in data.columns or 'phishing' not in data.columns:
        raise ValueError("Dataset must contain 'url' and 'phishing' columns.")
except FileNotFoundError:
    print(f"Error: File not found at {dataset_path}")
    exit()


processed_data = pdd_impl.extract_features_from_dataframe(data)

if processed_data.empty:
    print("No features extracted. Check dataset.")
    exit()

X = processed_data.drop(['labels'], axis=1)
y = processed_data['labels']

feature_names = X.columns.tolist()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)


print("Training Random Forest Classifier...")
rf_clf = RandomForestClassifier(n_estimators=10, random_state=42)
rf_clf.fit(X_train, y_train)


print("Evaluating...")
y_pred = rf_clf.predict(X_test)
y_probs = rf_clf.predict_proba(X_test)[:, 1]

print("\n--- Model Performance ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
try:
    print(f"ROC AUC:   {roc_auc_score(y_test, y_probs):.4f}")
except:
    print("ROC AUC:   N/A")


print("Saving model artifacts...")
joblib.dump(rf_clf, 'PhishFusion_model.pkl')
joblib.dump(feature_names, 'PhishFusion_feature_names.pkl')
print("Done. Model saved.")