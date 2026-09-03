# ensemble_classifier.py
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import lime
import lime.lime_tabular
import numpy as np

def get_stacking_model():
    """
    Returns the compiled Stacking Ensemble Model.
    Base Models: RF, LR, SVM
    Meta Model: LR
    """
    # Base Learners
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('lr', LogisticRegression(max_iter=1000, n_jobs=-1)),
        ('svm', SVC(probability=True, random_state=42, cache_size=1000)) 
    ]
    
    # Meta Learner
    final_estimator = LogisticRegression()
    
    # Stacking Classifier (Verified parameters for performance)
    clf = StackingClassifier(
        estimators=estimators, 
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1,  
        verbose=3   
    )
    return clf

def train_model(X_train, y_train):
    print("\n--- Training Stacking Ensemble ---")
    print("(This step includes SVM training, which may take time...)")
    model = get_stacking_model()
    model.fit(X_train, y_train)
    return model

def evaluate_metrics(y_true, y_pred, y_prob=None, label="Data"):
    """
    Computes Accuracy, Precision, Recall, F1, Confusion Matrix, AND ROC-AUC.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n--- Performance on {label} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # ROC-AUC Calculation
    roc_auc = "N/A"
    if y_prob is not None:
        try:
            # Check if binary (2 classes) or multiclass
            if len(np.unique(y_true)) == 2:
                # For binary, use the probability of the positive class (column 1)
                if y_prob.ndim == 2:
                    score = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    score = roc_auc_score(y_true, y_prob)
                print(f"ROC-AUC:   {score:.4f}")
                roc_auc = score
            else:
                # For multiclass, using 'ovr' (One-vs-Rest) strategy
                score = roc_auc_score(y_true, y_prob, multi_class='ovr')
                print(f"ROC-AUC:   {score:.4f}")
                roc_auc = score
        except Exception as e:
            print(f"ROC-AUC:   Could not calculate ({e})")

    print(f"Confusion Matrix:\n{cm}")
    
    return acc, prec, rec, f1, roc_auc

def explain_with_lime(model, X_train, feature_names, instance, num_features=25):
    """
    Generates LIME explanation for a single instance.
    """
    print("\n--- LIME Explanation ---")
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=['Legitimate', 'Phishing'],
            mode='classification',
            discretize_continuous=True
        )
        
        exp = explainer.explain_instance(
            data_row=instance, 
            predict_fn=model.predict_proba,
            num_features=num_features
        )
        
        # Print explanations
        print(exp.as_list())
        return exp
    except Exception as e:
        print(f"LIME Error: {e}")
        return None