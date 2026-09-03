import os
import subprocess
import sys
import logging
import time


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


def select_best_gpu():
   
    try:
        logging.info("Attempting to auto-select best GPU...")
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        
        memory_free = [int(x) for x in result.strip().split('\n')]
        
        if len(memory_free) > 0:
            best_gpu_index = memory_free.index(max(memory_free))
            max_mem = memory_free[best_gpu_index]
            
            logging.info(f"Found {len(memory_free)} GPUs. Free memory (MiB): {memory_free}")
            logging.info(f"Selecting GPU {best_gpu_index} (Free: {max_mem} MiB)")
            
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_index)
        else:
            logging.warning("nvidia-smi returned no GPUs.")
            
    except FileNotFoundError:
        logging.warning("nvidia-smi command not found. GPU selection skipped.")
    except Exception as e:
        logging.warning(f"Error selecting GPU: {e}. Defaulting to standard configuration.")

select_best_gpu()


import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, SpatialDropout1D, GlobalMaxPooling1D, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.info("Script started: CNN-Fusion Training")


try:
    logging.info("Loading dataset...")
    
    urls = pd.read_csv('/data/phishstorm.csv', encoding='ISO-8859-1')
    
    logging.info(f"Dataset loaded. Shape: {urls.shape}")

    y = urls['phishing']
    url_list = urls['url'].tolist()

    
    logging.info("Building vocabulary from URLs...")
    voc_chars = set(''.join(url_list))
    vocabulary = {x: idx + 1 for idx, x in enumerate(voc_chars)}
    logging.info(f"Vocabulary size: {len(vocabulary)} unique characters")


    with open('phishstorm_tokenizer.pickle', 'wb') as handle:
        pickle.dump(vocabulary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("Tokenizer saved to 'tokenizer.pickle'")

  
    logging.info("Preprocessing URLs (Char to Int conversion)...")
    X_indices = []
    for url in url_list:
        temp = [vocabulary[char] for char in url if char in vocabulary]
        X_indices.append(temp)

    max_url_len = 150
    logging.info(f"Padding sequences to length {max_url_len}...")
    X_pad = sequence.pad_sequences(X_indices, padding='post', maxlen=max_url_len)

   
    X_train, X_val, y_train, y_val = train_test_split(X_pad, y, test_size=0.1, random_state=7, stratify=y)
    logging.info(f"Data split completed. Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

    logging.info("Building CNN-Fusion model architecture...")
    inputs = Input(shape=(max_url_len,))

    
    embedding_layer = Embedding(len(vocabulary) + 1, 16)(inputs)

   
    conv1 = Conv1D(filters=128, kernel_size=8, activation='relu')(embedding_layer)
    drop1 = SpatialDropout1D(0.4)(conv1)
    pool1 = GlobalMaxPooling1D()(drop1)

    
    conv2 = Conv1D(filters=128, kernel_size=10, activation='relu')(embedding_layer)
    drop2 = SpatialDropout1D(0.4)(conv2)
    pool2 = GlobalMaxPooling1D()(drop2)

 
    conv3 = Conv1D(filters=256, kernel_size=12, activation='relu')(embedding_layer)
    drop3 = SpatialDropout1D(0.4)(conv3)
    pool3 = GlobalMaxPooling1D()(drop3)

    merged = concatenate([pool1, pool2, pool3])

    dense1 = Dense(128, activation='relu')(merged)
    drop_fc = Dropout(0.4)(dense1)
    outputs = Dense(1, activation='sigmoid')(drop_fc)

    model = Model(inputs=inputs, outputs=outputs)
    

    opt = Adam(learning_rate=0.001, epsilon=1e-08)
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    logging.info("Model compiled.")


    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                                patience=3, 
                                                verbose=1, 
                                                factor=0.8, 
                                                min_lr=0.00001)

    epochs = 50
    batch_size = 512

    logging.info(f"Starting training for {epochs} epochs with batch size {batch_size}...")
    start_time = time.time()
    
    history = model.fit(X_train, y_train,
                        batch_size=batch_size, 
                        epochs=epochs, 
                        verbose=1,
                        callbacks=[learning_rate_reduction],
                        validation_data=(X_val, y_val))

    duration = time.time() - start_time
    logging.info(f"Training completed in {duration:.2f} seconds.")

  
    logging.info("-" * 30)
    logging.info("CALCULATING VALIDATION METRICS")
    logging.info("-" * 30)
    
 
    y_val_prob = model.predict(X_val, verbose=1)
   
    y_val_pred = (y_val_prob > 0.5).astype(int)
    
    
    val_acc = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred)
    val_rec = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_prob)
    
  
    result_msg = (
        f"Validation Results:\n"
        f"Accuracy:  {val_acc:.4f}\n"
        f"Precision: {val_prec:.4f}\n"
        f"Recall:    {val_rec:.4f}\n"
        f"F1-Score:  {val_f1:.4f}\n"
        f"ROC AUC:   {val_auc:.4f}"
    )
    logging.info(result_msg)
    logging.info("-" * 30)

    # Save Model
    model.save('phishstorm_model.h5')
    logging.info("Model saved successfully")

except Exception as e:
    logging.error("An error occurred during training", exc_info=True)
    raise e