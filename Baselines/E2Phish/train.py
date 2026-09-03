import pandas as pd
import numpy as np
import joblib
import feature_extractor  
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score

# 1. Load Data
dataset_path = '/data/phishstorm.csv'
print(f"Loading Data from {dataset_path}...")

try:
    data = pd.read_csv(dataset_path, encoding='ISO-8859-1')
    # Ensure columns exist
    if 'url' not in data.columns or 'phishing' not in data.columns:
        raise ValueError("Dataset must contain 'url' and 'phishing' columns.")
except FileNotFoundError:
    print(f"Error: File not found at {dataset_path}")
    exit()


print("Extracting Features from URLs...")
processed_data = feature_extractor.extract_features_from_dataframe(data)

if processed_data.empty:
    raise ValueError("No features were extracted. Check your dataset.")


X_full = processed_data.drop(['labels'], axis=1)
y = processed_data['labels']


print("Calculating Mutual Information Scores...")

X_full = X_full.select_dtypes(include=[np.number])

discrete_features = X_full.dtypes == int
mi_scores = mutual_info_classif(X_full, y, discrete_features=discrete_features, random_state=42)
mi_scores = pd.Series(mi_scores, name='MI Scores', index=X_full.columns)
mi_scores = mi_scores.sort_values(ascending=False)


top_32_features = mi_scores.head(32).index.tolist()
print(f"Top 32 Features Selected: {top_32_features}")

X = X_full[top_32_features]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)


dt_clf = DecisionTreeClassifier(random_state=42)
gnb_clf = GaussianNB()

svm_clf = SVC(probability=True, random_state=42) 

base_classifiers = [
    ('dt', dt_clf),
    ('gnb', gnb_clf),
    ('svm', svm_clf)
]


meta_classifier = LogisticRegression(random_state=42)


print("Training E2Phish Stacking Ensemble...")
stacking_clf = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier)
stacking_clf.fit(X_train, y_train)


print("Evaluating Model...")
y_pred = stacking_clf.predict(X_test)
y_probs = stacking_clf.predict_proba(X_test)[:, 1] 

print("\n--- Model Performance (Validation Set) ---")
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

print("Saving model and feature list...")
joblib.dump(stacking_clf, 'phishstorm_e2phish_model.pkl')
joblib.dump(top_32_features, 'phishstorm_selected_features.pkl')
print("Training Complete. Model is saved.")