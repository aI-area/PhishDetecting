import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



TEST_DATA_PATH = '/data/verified_Phishtank.csv'  # CHANGE THIS to your new dataset filename
MODEL_PATH = 'PhishFuion_dephides_model.h5'
TOKENIZER_PATH = 'PhishFuion_tokenizer.pickle'
SEQUENCE_LENGTH = 200

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        exit()

    if 'url' not in df.columns or 'phishing' not in df.columns:
        print("Error: Dataset must have 'url' and 'phishing' columns.")
        exit()
        
    df['phishing'] = df['phishing'].astype(int)
    return df['url'].astype(str).tolist(), df['phishing'].values

if __name__ == "__main__":
 
    print("--- Loading Model and Tokenizer ---")
    try:
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("Model and Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        print("Please ensure you have run train.py first.")
        exit()

    print(f"--- Loading Test Data from {TEST_DATA_PATH} ---")
    urls, y_true = load_data(TEST_DATA_PATH)
    print(f"Loaded {len(urls)} test samples.")

  
    sequences = tokenizer.texts_to_sequences(urls)
    X_test = pad_sequences(sequences, maxlen=SEQUENCE_LENGTH, padding='post', truncating='post')

   
    print("--- Predicting ---")
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = (y_pred_probs > 0.5).astype(int)

 
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_probs)

   
    print("\n========================================")
    print("TEST RESULTS")
    print("========================================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print("========================================")