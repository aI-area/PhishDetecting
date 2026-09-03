import os
import subprocess
import sys
import logging
import pickle
import time
import pandas as pd
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("testing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


def select_best_gpu():
    """
    Detects GPUs, identifies the one with the most free memory,
    and sets CUDA_VISIBLE_DEVICES to that GPU index.
    """
    try:
        logging.info("Attempting to auto-select best GPU...")
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        memory_free = [int(x) for x in result.strip().split('\n')]
        
        if len(memory_free) > 0:
            best_gpu_index = memory_free.index(max(memory_free))
            logging.info(f"Selecting GPU {best_gpu_index} (Free: {memory_free[best_gpu_index]} MiB)")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_index)
        else:
            logging.warning("nvidia-smi returned no GPUs.")
            
    except Exception as e:
        logging.warning(f"GPU selection failed: {e}. Defaulting to standard config.")


select_best_gpu()

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.info("Script started: CNN-Fusion Testing")

try:

    logging.info("Loading saved model and tokenizer...")
    model = load_model('phishstorm_model.h5')
    
    with open('phishstorm_tokenizer.pickle', 'rb') as handle:
        vocabulary = pickle.load(handle)
    logging.info(f"Artifacts loaded. Vocabulary size: {len(vocabulary)}")

   
    new_data_path = '/data/ebbu2017.csv'
    
    logging.info(f"Reading new dataset from '{new_data_path}'...")
    
   
    df_new = pd.read_csv(new_data_path, encoding='ISO-8859-1')
    
  
    if 'url' not in df_new.columns or 'phishing' not in df_new.columns:
        logging.error("Dataset missing required columns ('url', 'phishing')")
        raise ValueError("Dataset must contain 'url' and 'phishing' columns.")
        
    logging.info(f"New dataset shape: {df_new.shape}")

    X_new_raw = df_new['url'].tolist()
    y_new = df_new['phishing'].values

    logging.info("Preprocessing new URLs...")
    X_new_indices = []
    skipped_chars = 0
    total_chars = 0
    
    for url in X_new_raw:
        temp = []
        for char in url:
            total_chars += 1
            
            if char in vocabulary:
                temp.append(vocabulary[char])
            else:
                skipped_chars += 1
        X_new_indices.append(temp)
    
    if skipped_chars > 0:
        logging.warning(f"Skipped {skipped_chars} characters ({(skipped_chars/total_chars)*100:.2f}%) not found in training vocabulary.")

    max_url_len = 150
    X_new_pad = sequence.pad_sequences(X_new_indices, padding='post', maxlen=max_url_len)
    logging.info("Padding completed.")


    logging.info("Starting prediction...")
    start_time = time.time()
    
    y_pred_prob = model.predict(X_new_pad, verbose=1)
    
 
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    duration = time.time() - start_time
    logging.info(f"Prediction completed in {duration:.2f} seconds.")

  
    acc = accuracy_score(y_new, y_pred)
    prec = precision_score(y_new, y_pred)
    rec = recall_score(y_new, y_pred)
    f1 = f1_score(y_new, y_pred)
    
    
    roc_auc = roc_auc_score(y_new, y_pred_prob)

   
    result_msg = (
        f"\n{'='*30}\n"
        f"TEST RESULTS ON NEW DATASET\n"
        f"{'='*30}\n"
        f"Accuracy:  {acc:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall:    {rec:.4f}\n"
        f"F1-Score:  {f1:.4f}\n"
        f"ROC AUC:   {roc_auc:.4f}\n"
        f"{'='*30}"
    )
    logging.info(result_msg)

except Exception as e:
    logging.error("An error occurred during testing", exc_info=True)
    raise e