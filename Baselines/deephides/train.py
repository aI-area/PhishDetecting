import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import sys
import warnings



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 


warnings.filterwarnings('ignore')

TRAIN_DATA_PATH = '/data/PhishFusion.csv'  # CHANGE THIS to your training dataset filename
MODEL_SAVE_PATH = 'PhishFuion_dephides_model.h5'
TOKENIZER_SAVE_PATH = 'PhishFuion_tokenizer.pickle'


SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 50
BATCH_SIZE = 128
EPOCHS = 20  
LEARNING_RATE = 0.001 
VAL_SPLIT = 0.2

def load_data(filepath):
  
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)

    if 'url' not in df.columns or 'phishing' not in df.columns:
        print("Error: Dataset must have 'url' and 'phishing' columns.")
        sys.exit(1)

  
    urls = df['url'].astype(str).tolist()
    labels = df['phishing'].astype(int).values
    return urls, labels

def build_cnn_complex(vocab_size):
  
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=EMBEDDING_DIM, input_length=SEQUENCE_LENGTH))

    # Layer 1
    model.add(Conv1D(128, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    # Layer 2
    model.add(Conv1D(128, 7, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    # Layer 3
    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    # Layer 4
    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    # Layer 5
    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    # Layer 6
    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    # Layer 7
    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

if __name__ == "__main__":
    print("--- Loading Data ---")
    urls, labels = load_data(TRAIN_DATA_PATH)


    print("--- Tokenizing ---")
 
    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(urls)
    
    sequences = tokenizer.texts_to_sequences(urls)
    X = pad_sequences(sequences, maxlen=SEQUENCE_LENGTH, padding='post', truncating='post')
    y = labels
    
    vocab_size = len(tokenizer.word_index)
    print(f"Vocabulary Size: {vocab_size}")


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SPLIT, random_state=42)
    print(f"Training samples: {len(X_train)} | Validation samples: {len(X_val)}")


    model = build_cnn_complex(vocab_size)
    
    
    optimizer = Adam(learning_rate=LEARNING_RATE) 
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

   
    print("--- Starting Training ---")
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val), verbose=1)

  
    print("\n--- Validation Metrics ---")
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int)

    val_acc = accuracy_score(y_val, y_pred)
    val_prec = precision_score(y_val, y_pred)
    val_rec = recall_score(y_val, y_pred)
    val_f1 = f1_score(y_val, y_pred)
    val_auc = roc_auc_score(y_val, y_pred_probs)

    print(f"Accuracy:  {val_acc:.4f}")
    print(f"Precision: {val_prec:.4f}")
    print(f"Recall:    {val_rec:.4f}")
    print(f"F1 Score:  {val_f1:.4f}")
    print(f"ROC AUC:   {val_auc:.4f}")

    # 6. Save
    model.save(MODEL_SAVE_PATH)
    with open(TOKENIZER_SAVE_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    print(f"Tokenizer saved to {TOKENIZER_SAVE_PATH}")