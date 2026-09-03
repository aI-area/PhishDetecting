#import os
#import warnings
#
## ----------------------------- Log Suppression -----------------------------
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL only
#os.environ["KMP_WARNINGS"] = "0"          # Turn off OpenMP warnings
#os.environ["KMP_AFFINITY"] = "disabled"   # Suppress KMP affinity logs
#
#
#warnings.filterwarnings("ignore")
#warnings.simplefilter(action='ignore', category=FutureWarning)
#
#import re
#import time
#import datetime
#import pdb
#import pickle
#import argparse
#import numpy as np
#import subprocess
#from tqdm import tqdm
#from bisect import bisect_left
#import tensorflow as tf
#
## Suppress TF internal logging
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#
#from tflearn.data_utils import to_categorical, pad_sequences
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
#from TextCNN import *
#from utils import *
#
## ----------------------------- GPU Selection -----------------------------
#def auto_select_gpu():
#    """Selects the GPU with the most available memory."""
#    try:
#        cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
#        output = subprocess.check_output(cmd.split())
#        lines = output.decode('utf-8').strip().split('\n')
#        
#        gpu_stats = []
#        for line in lines:
#            if not line: continue
#            idx, mem = line.split(',')
#            gpu_stats.append((int(idx.strip()), int(mem.strip())))
#        
#        if gpu_stats:
#            gpu_stats.sort(key=lambda x: x[1], reverse=True)
#            best_gpu_id = gpu_stats[0][0]
#            print("[GPU Setup] Auto-selecting GPU ID {} with {} MiB free.".format(best_gpu_id, gpu_stats[0][1]))
#            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_id)
#        else:
#            print("[GPU Setup] No GPUs found via nvidia-smi.")
#    except Exception as e:
#        print("[GPU Setup] GPU selection failed: {}. Using default.".format(e))
#
#auto_select_gpu()
#
## ----------------------------- Argparser -----------------------------
#parser = argparse.ArgumentParser(description="Train URLNet model")
#
## data args
#parser.add_argument('--data.max_len_words', type=int, default=200, metavar="MLW")
#parser.add_argument('--data.max_len_chars', type=int, default=200, metavar="MLC")
#parser.add_argument('--data.max_len_subwords', type=int, default=20, metavar="MLSW")
#parser.add_argument('--data.min_word_freq', type=int, default=1, metavar="MWF")
#parser.add_argument('--data.dev_pct', type=float, default=0.001, metavar="DEVPCT")
#parser.add_argument('--data.data_dir', type=str, default='train_10000.txt', metavar="DATADIR")
#parser.add_argument("--data.delimit_mode", type=int, default=1, metavar="DLMODE")
#
## model args
#parser.add_argument('--model.emb_dim', type=int, default=32, metavar="EMBDIM")
#parser.add_argument('--model.filter_sizes', type=str, default="3,4,5,6", metavar="FILTERSIZES")
#parser.add_argument('--model.emb_mode', type=int, default=1, metavar="EMBMODE")
#
## train args
#parser.add_argument('--train.nb_epochs', type=int, default=10, metavar="NEPOCHS")
#parser.add_argument('--train.batch_size', type=int, default=128, metavar="BATCHSIZE")
#parser.add_argument('--train.l2_reg_lambda', type=float, default=0.0, metavar="L2LREGLAMBDA")
#parser.add_argument('--train.lr', type=float, default=0.001, metavar="LR")
#
## log args
#parser.add_argument('--log.output_dir', type=str, default="runs/10000/", metavar="OUTPUTDIR")
#parser.add_argument('--log.print_every', type=int, default=50, metavar="PRINTEVERY")
#parser.add_argument('--log.eval_every', type=int, default=500, metavar="EVALEVERY")
#parser.add_argument('--log.checkpoint_every', type=int, default=500, metavar="CHECKPOINTEVERY")
#
#FLAGS = vars(parser.parse_args())
#
## ---------------- CSV auto-conversion ----------------
#import csv, tempfile
#from pathlib import Path
#
#def _csv_to_urlnet_txt(input_csv, url_col_name="url", label_col_name="phishing"):
#    input_csv = Path(input_csv)
#    out_path = Path(tempfile.gettempdir()) / "{}_urlnet.txt".format(input_csv.stem)
#    
#    with input_csv.open("r", newline="", encoding="utf-8") as f_in:
#        reader = csv.DictReader(f_in)
#        field_map = {name.lower(): name for name in reader.fieldnames}
#        if url_col_name not in field_map or label_col_name not in field_map:
#            raise ValueError(f"CSV missing columns. Found: {reader.fieldnames}")
#        
#        url_col = field_map[url_col_name]
#        lab_col = field_map[label_col_name]
#        
#        with out_path.open("w", newline="\n", encoding="utf-8") as f_out:
#            n_written = 0
#            for row in reader:
#                url = (row[url_col] or "").strip()
#                try:
#                    lab = int(row[lab_col])
#                except: continue
#                
#                mapped = "+1" if lab == 1 else "-1"
#                if url:
#                    f_out.write("{}\t{}\n".format(mapped, url))
#                    n_written += 1
#    
#    print("[convert] Wrote {} lines to {}".format(n_written, out_path))
#    return str(out_path)
#
#if str(FLAGS["data.data_dir"]).lower().endswith(".csv"):
#    print("[convert] Detected CSV. Converting...")
#    FLAGS["data.data_dir"] = _csv_to_urlnet_txt(FLAGS["data.data_dir"], "url", "phishing")
#
## ---------------- Load & preprocess data ----------------
#urls, labels = read_data(FLAGS["data.data_dir"])
#
#high_freq_words = None
#if FLAGS["data.min_word_freq"] > 0:
#    x1, word_reverse_dict = get_word_vocab(urls, FLAGS["data.max_len_words"], FLAGS["data.min_word_freq"])
#    high_freq_words = sorted(list(word_reverse_dict.values()))
#
#x, word_reverse_dict = get_word_vocab(urls, FLAGS["data.max_len_words"])
#word_x = get_words(x, word_reverse_dict, FLAGS["data.delimit_mode"], urls)
#ngramed_id_x, ngrams_dict, worded_id_x, words_dict = ngram_id_x(
#    word_x, FLAGS["data.max_len_subwords"], high_freq_words
#)
#
#chars_dict = ngrams_dict
#chared_id_x = char_id_x(urls, chars_dict, FLAGS["data.max_len_chars"])
#
#pos_x, neg_x = [], []
#for i in range(len(labels)):
#    if labels[i] == 1: pos_x.append(i)
#    else: neg_x.append(i)
#
#x_train, y_train, x_test, y_test = prep_train_test(np.array(pos_x), np.array(neg_x), FLAGS["data.dev_pct"])
#
#x_train_char = get_ngramed_id_x(x_train, ngramed_id_x)
#x_test_char  = get_ngramed_id_x(x_test,  ngramed_id_x)
#x_train_word = get_ngramed_id_x(x_train, worded_id_x)
#x_test_word  = get_ngramed_id_x(x_test,  worded_id_x)
#x_train_char_seq = get_ngramed_id_x(x_train, chared_id_x)
#x_test_char_seq  = get_ngramed_id_x(x_test,  chared_id_x)
#
## ---------------- Helper: Softmax ----------------
#def softmax_numpy(x):
#    """Compute softmax values for each sets of scores in x."""
#    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
#    return e_x / np.sum(e_x, axis=1, keepdims=True)
#
## ---------------- Training helpers ----------------
#def train_dev_step(x, y, emb_mode, is_train=True):
#    p = 0.5 if is_train else 1.0
#    feed_dict = {cnn.input_y: y, cnn.dropout_keep_prob: p}
#    
#    # Map inputs based on embedding mode
#    if emb_mode == 1:
#        feed_dict[cnn.input_x_char_seq] = x[0]
#    elif emb_mode == 2:
#        feed_dict[cnn.input_x_word] = x[0]
#    elif emb_mode == 3:
#        feed_dict[cnn.input_x_char_seq] = x[0]
#        feed_dict[cnn.input_x_word] = x[1]
#    elif emb_mode == 4:
#        feed_dict[cnn.input_x_word] = x[0]
#        feed_dict[cnn.input_x_char] = x[1]
#        feed_dict[cnn.input_x_char_pad_idx] = x[2]
#    elif emb_mode == 5:
#        feed_dict[cnn.input_x_char_seq] = x[0]
#        feed_dict[cnn.input_x_word] = x[1]
#        feed_dict[cnn.input_x_char] = x[2]
#        feed_dict[cnn.input_x_char_pad_idx] = x[3]
#    
#    if is_train:
#        _, step, loss, acc = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
#        return step, loss, acc, None
#    else:
#        # ROBUST PROBABILITY FETCHING
#        if hasattr(cnn, 'probs'):
#            step, loss, acc, probs = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.probs], feed_dict)
#        elif hasattr(cnn, 'scores'):
#            step, loss, acc, scores = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.scores], feed_dict)
#            probs = softmax_numpy(scores)
#        else:
#            step, loss, acc = sess.run([global_step, cnn.loss, cnn.accuracy], feed_dict)
#            probs = None
#        return step, loss, acc, probs
#
#def make_batches(x_train_char_seq, x_train_word, x_train_char, y_train, batch_size, nb_epochs, shuffle=False):
#    # Pack data based on mode
#    if FLAGS["model.emb_mode"] == 1:
#        batch_data = list(zip(x_train_char_seq, y_train))
#    elif FLAGS["model.emb_mode"] == 2:
#        batch_data = list(zip(x_train_word, y_train))
#    elif FLAGS["model.emb_mode"] == 3:
#        batch_data = list(zip(x_train_char_seq, x_train_word, y_train))
#    elif FLAGS["model.emb_mode"] == 4:
#        batch_data = list(zip(x_train_char, x_train_word, y_train))
#    elif FLAGS["model.emb_mode"] == 5:
#        batch_data = list(zip(x_train_char, x_train_word, x_train_char_seq, y_train))
#
#    batches = batch_iter(batch_data, batch_size, nb_epochs, shuffle)
#
#    if nb_epochs > 1:
#        nb_batches_per_epoch = int(len(batch_data) / batch_size)
#        if len(batch_data) % batch_size != 0: nb_batches_per_epoch += 1
#        return batches, nb_batches_per_epoch, int(nb_batches_per_epoch * nb_epochs)
#    else:
#        return batches
#
#def prep_batches(batch):
#    # Unpack based on mode
#    if FLAGS["model.emb_mode"] == 1:
#        x_char_seq, y_batch = zip(*batch)
#    elif FLAGS["model.emb_mode"] == 2:
#        x_word, y_batch = zip(*batch)
#    elif FLAGS["model.emb_mode"] == 3:
#        x_char_seq, x_word, y_batch = zip(*batch)
#    elif FLAGS["model.emb_mode"] == 4:
#        x_char, x_word, y_batch = zip(*batch)
#    elif FLAGS["model.emb_mode"] == 5:
#        x_char, x_word, x_char_seq, y_batch = zip(*batch)
#
#    x_batch = []
#    # Padding logic
#    if FLAGS["model.emb_mode"] in [1, 3, 5]:
#        x_batch.append(pad_seq_in_word(x_char_seq, FLAGS["data.max_len_chars"]))
#    if FLAGS["model.emb_mode"] in [2, 3, 4, 5]:
#        x_batch.append(pad_seq_in_word(x_word, FLAGS["data.max_len_words"]))
#    if FLAGS["model.emb_mode"] in [4, 5]:
#        x_c, x_c_pad = pad_seq(x_char, FLAGS["data.max_len_words"], FLAGS["data.max_len_subwords"], FLAGS["model.emb_dim"])
#        x_batch.extend([x_c, x_c_pad])
#    return x_batch, y_batch
#
## ---------------- Build & Train ----------------
#with tf.Graph().as_default():
#    session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#    session_conf.gpu_options.allow_growth = True
#    sess = tf.compat.v1.Session(config=session_conf)
#
#    with sess.as_default():
#        cnn = TextCNN(
#            char_ngram_vocab_size=len(ngrams_dict) + 1,
#            word_ngram_vocab_size=len(words_dict) + 1,
#            char_vocab_size=len(chars_dict) + 1,
#            embedding_size=FLAGS["model.emb_dim"],
#            word_seq_len=FLAGS["data.max_len_words"],
#            char_seq_len=FLAGS["data.max_len_chars"],
#            l2_reg_lambda=FLAGS["train.l2_reg_lambda"],
#            mode=FLAGS["model.emb_mode"],
#            filter_sizes=list(map(int, FLAGS["model.filter_sizes"].split(",")))
#        )
#
#        global_step = tf.Variable(0, name="global_step", trainable=False)
#        optimizer = tf.compat.v1.train.AdamOptimizer(FLAGS["train.lr"])
#        grads_and_vars = optimizer.compute_gradients(cnn.loss)
#        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
#
#        if not os.path.exists(FLAGS["log.output_dir"]):
#            os.makedirs(FLAGS["log.output_dir"])
#
#        # Save dictionaries
#        pickle.dump(ngrams_dict, open(os.path.join(FLAGS["log.output_dir"], "subwords_dict.p"), "wb"))
#        pickle.dump(words_dict, open(os.path.join(FLAGS["log.output_dir"], "words_dict.p"), "wb"))
#        pickle.dump(chars_dict, open(os.path.join(FLAGS["log.output_dir"], "chars_dict.p"), "wb"))
#
#        # Logs
#        train_log_dir = os.path.join(FLAGS["log.output_dir"], "train_logs.csv")
#        with open(train_log_dir, "w") as f: f.write("step,time,loss,acc\n")
#        
#        val_log_dir = os.path.join(FLAGS["log.output_dir"], "val_logs.csv")
#        with open(val_log_dir, "w") as f: f.write("step,time,loss,acc\n")
#
#        # Saver
#        checkpoint_prefix = os.path.join(FLAGS["log.output_dir"], "checkpoints", "model")
#        if not os.path.exists(os.path.dirname(checkpoint_prefix)): os.makedirs(os.path.dirname(checkpoint_prefix))
#        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)
#
#        sess.run(tf.compat.v1.global_variables_initializer())
#
#        train_batches, nb_batches_per_epoch, nb_batches = make_batches(
#            x_train_char_seq, x_train_word, x_train_char, y_train,
#            FLAGS["train.batch_size"], FLAGS['train.nb_epochs'], True
#        )
#
#        min_dev_loss = float('Inf')
#        dev_loss = float('Inf')
#        dev_acc = 0.0
#
#        it = tqdm(range(nb_batches), desc="Training", ncols=0)
#
#        for idx in it:
#            batch = next(train_batches)
#            x_batch, y_batch = prep_batches(batch)
#            step, loss, acc, _ = train_dev_step(x_batch, y_batch, emb_mode=FLAGS["model.emb_mode"], is_train=True)
#
#            if step % FLAGS["log.print_every"] == 0:
#                with open(train_log_dir, "a") as f:
#                    f.write("{:d},{:s},{:e},{:e}\n".format(step, datetime.datetime.now().isoformat(), loss, acc))
#                it.set_postfix(trn_loss='{:.3e}'.format(loss), trn_acc='{:.3e}'.format(acc))
#
#            if step % FLAGS["log.eval_every"] == 0 or idx == (nb_batches - 1):
#                # Standard Dev Evaluation
#                test_batches_dev = make_batches(x_test_char_seq, x_test_word, x_test_char, y_test, FLAGS['train.batch_size'], 1, False)
#                total_loss, nb_corrects, nb_instances = 0.0, 0.0, 0
#                
#                for test_batch in test_batches_dev:
#                    x_dev, y_dev = prep_batches(test_batch)
#                    _, batch_loss, batch_acc, _ = train_dev_step(x_dev, y_dev, emb_mode=FLAGS["model.emb_mode"], is_train=False)
#                    n = x_dev[0].shape[0]
#                    total_loss += batch_loss * n
#                    nb_corrects += batch_acc * n
#                    nb_instances += n
#                
#                dev_loss = total_loss / max(nb_instances, 1)
#                dev_acc = nb_corrects / max(nb_instances, 1)
#                
#                with open(val_log_dir, "a") as f:
#                    f.write("{:d},{:s},{:e},{:e}\n".format(step, datetime.datetime.now().isoformat(), dev_loss, dev_acc))
#
#                if dev_loss < min_dev_loss:
#                    path = saver.save(sess, checkpoint_prefix, global_step=step)
#                    min_dev_loss = dev_loss
#
#        # =========================================================================
#        #                         FINAL TEST SET EVALUATION
#        # =========================================================================
#        print("\n" + "="*50)
#        print(" TRAINING COMPLETE. COMPUTING FINAL METRICS ON TEST SET")
#        print("="*50)
#
#        final_test_batches = make_batches(x_test_char_seq, x_test_word, x_test_char, y_test, FLAGS['train.batch_size'], 1, False)
#        
#        all_true = []
#        all_pred = []
#        all_prob = [] 
#
#        for batch in final_test_batches:
#            x_b, y_b = prep_batches(batch)
#            _, _, _, probs_b = train_dev_step(x_b, y_b, emb_mode=FLAGS["model.emb_mode"], is_train=False)
#            
#            all_true.extend(y_b)
#            if probs_b is not None:
#                all_prob.extend(probs_b[:, 1])
#                all_pred.extend(np.argmax(probs_b, axis=1))
#            else:
#                # Fallback
#                all_pred.extend([0]*len(y_b))
#                all_prob.extend([0.0]*len(y_b))
#        
#        # Calculate Scikit-Learn Metrics
#        try:
#            y_true_np = np.array(all_true)
#            
#            # --- CRITICAL FIX: Handle One-Hot Encoding for Scikit-Learn ---
#            # If y_true is (N, 2), convert to (N,) class indices
#            if y_true_np.ndim > 1 and y_true_np.shape[1] > 1:
#                y_true_np = np.argmax(y_true_np, axis=1)
#            # ---------------------------------------------------------------
#            
#            y_pred_np = np.array(all_pred)
#            
#            if len(y_pred_np) == 0:
#                 print("Error: No predictions generated. Check model output.")
#            else:
#                final_acc = accuracy_score(y_true_np, y_pred_np)
#                final_pre = precision_score(y_true_np, y_pred_np, average='binary', zero_division=0)
#                final_rec = recall_score(y_true_np, y_pred_np, average='binary', zero_division=0)
#                final_f1 = f1_score(y_true_np, y_pred_np, average='binary', zero_division=0)
#                
#                if len(all_prob) > 0:
#                    try:
#                        final_auc = roc_auc_score(y_true_np, all_prob)
#                    except ValueError:
#                        final_auc = 0.0 # Handle cases with only one class in test set
#                else:
#                    final_auc = 0.0
#
#                print("{:<15} {:.4f}".format("Accuracy:", final_acc))
#                print("{:<15} {:.4f}".format("Precision:", final_pre))
#                print("{:<15} {:.4f}".format("Recall:", final_rec))
#                print("{:<15} {:.4f}".format("F1-Score:", final_f1))
#                print("{:<15} {:.4f}".format("ROC AUC:", final_auc))
#                print("="*50)
#
#                with open(os.path.join(FLAGS["log.output_dir"], "final_metrics.txt"), "w") as f:
#                    f.write(f"Accuracy: {final_acc}\nPrecision: {final_pre}\nRecall: {final_rec}\nF1: {final_f1}\nAUC: {final_auc}\n")
#
#        except Exception as e:
#            print(f"Error computing final metrics: {e}")
#

import os
import warnings


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ["KMP_WARNINGS"] = "0"          
os.environ["KMP_AFFINITY"] = "disabled"   


warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import re
import time
import datetime
import pdb
import pickle
import argparse
import numpy as np
import subprocess
from tqdm import tqdm
from bisect import bisect_left
import tensorflow as tf

# Suppress TF internal logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from TextCNN import *
from utils import *


def auto_select_gpu():
    """Selects the GPU with the most available memory."""
    try:
        cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd.split())
        lines = output.decode('utf-8').strip().split('\n')
        
        gpu_stats = []
        for line in lines:
            if not line: continue
            idx, mem = line.split(',')
            gpu_stats.append((int(idx.strip()), int(mem.strip())))
        
        if gpu_stats:
            gpu_stats.sort(key=lambda x: x[1], reverse=True)
            best_gpu_id = gpu_stats[0][0]
            print("[GPU Setup] Auto-selecting GPU ID {} with {} MiB free.".format(best_gpu_id, gpu_stats[0][1]))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_id)
        else:
            print("[GPU Setup] No GPUs found via nvidia-smi.")
    except Exception as e:
        print("[GPU Setup] GPU selection failed: {}. Using default.".format(e))

auto_select_gpu()


parser = argparse.ArgumentParser(description="Train URLNet model")

# data args
parser.add_argument('--data.max_len_words', type=int, default=200, metavar="MLW")
parser.add_argument('--data.max_len_chars', type=int, default=200, metavar="MLC")
parser.add_argument('--data.max_len_subwords', type=int, default=20, metavar="MLSW")
parser.add_argument('--data.min_word_freq', type=int, default=1, metavar="MWF")
parser.add_argument('--data.dev_pct', type=float, default=0.001, metavar="DEVPCT")
parser.add_argument('--data.data_dir', type=str, default='train_10000.txt', metavar="DATADIR")
parser.add_argument("--data.delimit_mode", type=int, default=1, metavar="DLMODE")

# model args
parser.add_argument('--model.emb_dim', type=int, default=32, metavar="EMBDIM")
parser.add_argument('--model.filter_sizes', type=str, default="3,4,5,6", metavar="FILTERSIZES")
parser.add_argument('--model.emb_mode', type=int, default=1, metavar="EMBMODE")

# train args
parser.add_argument('--train.nb_epochs', type=int, default=10, metavar="NEPOCHS")
parser.add_argument('--train.batch_size', type=int, default=128, metavar="BATCHSIZE")
parser.add_argument('--train.l2_reg_lambda', type=float, default=0.0, metavar="L2LREGLAMBDA")
parser.add_argument('--train.lr', type=float, default=0.001, metavar="LR")

# log args
parser.add_argument('--log.output_dir', type=str, default="runs/10000/", metavar="OUTPUTDIR")
parser.add_argument('--log.print_every', type=int, default=50, metavar="PRINTEVERY")
parser.add_argument('--log.eval_every', type=int, default=500, metavar="EVALEVERY")
parser.add_argument('--log.checkpoint_every', type=int, default=500, metavar="CHECKPOINTEVERY")

FLAGS = vars(parser.parse_args())


import csv, tempfile
from pathlib import Path

def _csv_to_urlnet_txt(input_csv, url_col_name="url", label_col_name="phishing"):
    input_csv = Path(input_csv)
    out_path = Path(tempfile.gettempdir()) / "{}_urlnet.txt".format(input_csv.stem)
    
    with input_csv.open("r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        field_map = {name.lower(): name for name in reader.fieldnames}
        if url_col_name not in field_map or label_col_name not in field_map:
            raise ValueError(f"CSV missing columns. Found: {reader.fieldnames}")
        
        url_col = field_map[url_col_name]
        lab_col = field_map[label_col_name]
        
        with out_path.open("w", newline="\n", encoding="utf-8") as f_out:
            n_written = 0
            for row in reader:
                url = (row[url_col] or "").strip()
                try:
                    lab = int(row[lab_col])
                except: continue
                
                mapped = "+1" if lab == 1 else "-1"
                if url:
                    f_out.write("{}\t{}\n".format(mapped, url))
                    n_written += 1
    
    print("[convert] Wrote {} lines to {}".format(n_written, out_path))
    return str(out_path)

if str(FLAGS["data.data_dir"]).lower().endswith(".csv"):
    print("[convert] Detected CSV. Converting...")
    FLAGS["data.data_dir"] = _csv_to_urlnet_txt(FLAGS["data.data_dir"], "url", "phishing")


urls, labels = read_data(FLAGS["data.data_dir"])

high_freq_words = None
if FLAGS["data.min_word_freq"] > 0:
    x1, word_reverse_dict = get_word_vocab(urls, FLAGS["data.max_len_words"], FLAGS["data.min_word_freq"])
    high_freq_words = sorted(list(word_reverse_dict.values()))

x, word_reverse_dict = get_word_vocab(urls, FLAGS["data.max_len_words"])
word_x = get_words(x, word_reverse_dict, FLAGS["data.delimit_mode"], urls)
ngramed_id_x, ngrams_dict, worded_id_x, words_dict = ngram_id_x(
    word_x, FLAGS["data.max_len_subwords"], high_freq_words
)

chars_dict = ngrams_dict
chared_id_x = char_id_x(urls, chars_dict, FLAGS["data.max_len_chars"])

pos_x, neg_x = [], []
for i in range(len(labels)):
    if labels[i] == 1: pos_x.append(i)
    else: neg_x.append(i)

x_train, y_train, x_test, y_test = prep_train_test(np.array(pos_x), np.array(neg_x), FLAGS["data.dev_pct"])

x_train_char = get_ngramed_id_x(x_train, ngramed_id_x)
x_test_char  = get_ngramed_id_x(x_test,  ngramed_id_x)
x_train_word = get_ngramed_id_x(x_train, worded_id_x)
x_test_word  = get_ngramed_id_x(x_test,  worded_id_x)
x_train_char_seq = get_ngramed_id_x(x_train, chared_id_x)
x_test_char_seq  = get_ngramed_id_x(x_test,  chared_id_x)


def softmax_numpy(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def train_dev_step(x, y, emb_mode, is_train=True):
    p = 0.5 if is_train else 1.0
    feed_dict = {cnn.input_y: y, cnn.dropout_keep_prob: p}
   
    if emb_mode == 1:
        feed_dict[cnn.input_x_char_seq] = x[0]
    elif emb_mode == 2:
        feed_dict[cnn.input_x_word] = x[0]
    elif emb_mode == 3:
        feed_dict[cnn.input_x_char_seq] = x[0]
        feed_dict[cnn.input_x_word] = x[1]
    elif emb_mode == 4:
        feed_dict[cnn.input_x_word] = x[0]
        feed_dict[cnn.input_x_char] = x[1]
        feed_dict[cnn.input_x_char_pad_idx] = x[2]
    elif emb_mode == 5:
        feed_dict[cnn.input_x_char_seq] = x[0]
        feed_dict[cnn.input_x_word] = x[1]
        feed_dict[cnn.input_x_char] = x[2]
        feed_dict[cnn.input_x_char_pad_idx] = x[3]
    
    if is_train:
        _, step, loss, acc = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
        return step, loss, acc, None
    else:
     
        if hasattr(cnn, 'probs'):
            step, loss, acc, probs = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.probs], feed_dict)
        elif hasattr(cnn, 'scores'):
            step, loss, acc, scores = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.scores], feed_dict)
            probs = softmax_numpy(scores)
        else:
            step, loss, acc = sess.run([global_step, cnn.loss, cnn.accuracy], feed_dict)
            probs = None
        return step, loss, acc, probs

def make_batches(x_train_char_seq, x_train_word, x_train_char, y_train, batch_size, nb_epochs, shuffle=False):

    if FLAGS["model.emb_mode"] == 1:
        batch_data = list(zip(x_train_char_seq, y_train))
    elif FLAGS["model.emb_mode"] == 2:
        batch_data = list(zip(x_train_word, y_train))
    elif FLAGS["model.emb_mode"] == 3:
        batch_data = list(zip(x_train_char_seq, x_train_word, y_train))
    elif FLAGS["model.emb_mode"] == 4:
        batch_data = list(zip(x_train_char, x_train_word, y_train))
    elif FLAGS["model.emb_mode"] == 5:
        batch_data = list(zip(x_train_char, x_train_word, x_train_char_seq, y_train))

    batches = batch_iter(batch_data, batch_size, nb_epochs, shuffle)

    if nb_epochs > 1:
        nb_batches_per_epoch = int(len(batch_data) / batch_size)
        if len(batch_data) % batch_size != 0: nb_batches_per_epoch += 1
        return batches, nb_batches_per_epoch, int(nb_batches_per_epoch * nb_epochs)
    else:
        return batches

def prep_batches(batch):
    
    if FLAGS["model.emb_mode"] == 1:
        x_char_seq, y_batch = zip(*batch)
    elif FLAGS["model.emb_mode"] == 2:
        x_word, y_batch = zip(*batch)
    elif FLAGS["model.emb_mode"] == 3:
        x_char_seq, x_word, y_batch = zip(*batch)
    elif FLAGS["model.emb_mode"] == 4:
        x_char, x_word, y_batch = zip(*batch)
    elif FLAGS["model.emb_mode"] == 5:
        x_char, x_word, x_char_seq, y_batch = zip(*batch)

    x_batch = []
 
    if FLAGS["model.emb_mode"] in [1, 3, 5]:
        x_batch.append(pad_seq_in_word(x_char_seq, FLAGS["data.max_len_chars"]))
    if FLAGS["model.emb_mode"] in [2, 3, 4, 5]:
        x_batch.append(pad_seq_in_word(x_word, FLAGS["data.max_len_words"]))
    if FLAGS["model.emb_mode"] in [4, 5]:
        x_c, x_c_pad = pad_seq(x_char, FLAGS["data.max_len_words"], FLAGS["data.max_len_subwords"], FLAGS["model.emb_dim"])
        x_batch.extend([x_c, x_c_pad])
    return x_batch, y_batch


with tf.Graph().as_default():
    session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=session_conf)

    with sess.as_default():
        cnn = TextCNN(
            char_ngram_vocab_size=len(ngrams_dict) + 1,
            word_ngram_vocab_size=len(words_dict) + 1,
            char_vocab_size=len(chars_dict) + 1,
            embedding_size=FLAGS["model.emb_dim"],
            word_seq_len=FLAGS["data.max_len_words"],
            char_seq_len=FLAGS["data.max_len_chars"],
            l2_reg_lambda=FLAGS["train.l2_reg_lambda"],
            mode=FLAGS["model.emb_mode"],
            filter_sizes=list(map(int, FLAGS["model.filter_sizes"].split(",")))
        )

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.compat.v1.train.AdamOptimizer(FLAGS["train.lr"])
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        if not os.path.exists(FLAGS["log.output_dir"]):
            os.makedirs(FLAGS["log.output_dir"])

        # Save dictionaries
        pickle.dump(ngrams_dict, open(os.path.join(FLAGS["log.output_dir"], "subwords_dict.p"), "wb"))
        pickle.dump(words_dict, open(os.path.join(FLAGS["log.output_dir"], "words_dict.p"), "wb"))
        pickle.dump(chars_dict, open(os.path.join(FLAGS["log.output_dir"], "chars_dict.p"), "wb"))

   
        train_log_dir = os.path.join(FLAGS["log.output_dir"], "train_logs.csv")
        with open(train_log_dir, "w") as f: f.write("step,time,loss,acc\n")
        
        val_log_dir = os.path.join(FLAGS["log.output_dir"], "val_logs.csv")
        with open(val_log_dir, "w") as f: f.write("step,time,loss,acc\n")

       
        checkpoint_prefix = os.path.join(FLAGS["log.output_dir"], "checkpoints", "model")
        if not os.path.exists(os.path.dirname(checkpoint_prefix)): os.makedirs(os.path.dirname(checkpoint_prefix))
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)

        sess.run(tf.compat.v1.global_variables_initializer())

        train_batches, nb_batches_per_epoch, nb_batches = make_batches(
            x_train_char_seq, x_train_word, x_train_char, y_train,
            FLAGS["train.batch_size"], FLAGS['train.nb_epochs'], True
        )

        min_dev_loss = float('Inf')
        dev_loss = float('Inf')
        dev_acc = 0.0

        it = tqdm(range(nb_batches), desc="Training", ncols=0)

        for idx in it:
            batch = next(train_batches)
            x_batch, y_batch = prep_batches(batch)
            step, loss, acc, _ = train_dev_step(x_batch, y_batch, emb_mode=FLAGS["model.emb_mode"], is_train=True)

            if step % FLAGS["log.print_every"] == 0:
                with open(train_log_dir, "a") as f:
                    f.write("{:d},{:s},{:e},{:e}\n".format(step, datetime.datetime.now().isoformat(), loss, acc))
                it.set_postfix(trn_loss='{:.3e}'.format(loss), trn_acc='{:.3e}'.format(acc))

            if step % FLAGS["log.eval_every"] == 0 or idx == (nb_batches - 1):
                # Standard Dev Evaluation
                test_batches_dev = make_batches(x_test_char_seq, x_test_word, x_test_char, y_test, FLAGS['train.batch_size'], 1, False)
                total_loss, nb_corrects, nb_instances = 0.0, 0.0, 0
                
                for test_batch in test_batches_dev:
                    x_dev, y_dev = prep_batches(test_batch)
                    _, batch_loss, batch_acc, _ = train_dev_step(x_dev, y_dev, emb_mode=FLAGS["model.emb_mode"], is_train=False)
                    n = x_dev[0].shape[0]
                    total_loss += batch_loss * n
                    nb_corrects += batch_acc * n
                    nb_instances += n
                
                dev_loss = total_loss / max(nb_instances, 1)
                dev_acc = nb_corrects / max(nb_instances, 1)
                
                with open(val_log_dir, "a") as f:
                    f.write("{:d},{:s},{:e},{:e}\n".format(step, datetime.datetime.now().isoformat(), dev_loss, dev_acc))

                if dev_loss < min_dev_loss:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    min_dev_loss = dev_loss

   
        print("\n" + "="*50)
        print(" TRAINING COMPLETE. COMPUTING FINAL METRICS ON TEST SET")
        print("="*50)

        final_test_batches = make_batches(
            x_test_char_seq, x_test_word, x_test_char, y_test, 
            FLAGS['train.batch_size'], 1, False
        )
        
        all_true = []
        all_pred = []
        all_prob = [] 

        for batch in final_test_batches:
            x_b, y_b = prep_batches(batch)
            _, _, _, probs_b = train_dev_step(x_b, y_b, emb_mode=FLAGS["model.emb_mode"], is_train=False)
            
            all_true.extend(y_b)
            if probs_b is not None:
              
                all_prob.extend(probs_b[:, 1])
             
                all_pred.extend(np.argmax(probs_b, axis=1))
            else:
              
                all_pred.extend([0]*len(y_b))
                all_prob.extend([0.0]*len(y_b))
        
        try:
       
            y_true_np = np.array(all_true)
            y_pred_np = np.array(all_pred)
            
          
            if y_true_np.ndim > 1 and y_true_np.shape[1] > 1:
                y_true_np = np.argmax(y_true_np, axis=1)

    
            tp = np.sum((y_true_np == 1) & (y_pred_np == 1))
            tn = np.sum((y_true_np == 0) & (y_pred_np == 0))
            fp = np.sum((y_true_np == 0) & (y_pred_np == 1))
            fn = np.sum((y_true_np == 1) & (y_pred_np == 0))

            # 3. Standard Metrics
            final_acc = accuracy_score(y_true_np, y_pred_np)
            final_pre = precision_score(y_true_np, y_pred_np, average='binary', zero_division=0)
            final_rec = recall_score(y_true_np, y_pred_np, average='binary', zero_division=0)
            final_f1 = f1_score(y_true_np, y_pred_np, average='binary', zero_division=0)
            
            final_auc = 0.0
            if len(all_prob) > 0 and len(np.unique(y_true_np)) > 1:
                try:
                    final_auc = roc_auc_score(y_true_np, all_prob)
                except ValueError:
                    final_auc = 0.0 

            print("-" * 30)
            print(f"Total Test Samples: {len(y_true_np)}")
            print(f"Confusion Matrix: TP={tp}  FP={fp}  TN={tn}  FN={fn}")
            print("-" * 30)
            print("{:<15} {:.4f}".format("Accuracy:", final_acc))
            print("{:<15} {:.4f}".format("Precision:", final_pre))
            print("{:<15} {:.4f}".format("Recall:", final_rec))
            print("{:<15} {:.4f}".format("F1-Score:", final_f1))
            print("{:<15} {:.4f}".format("ROC AUC:", final_auc))
            print("="*50)

            with open(os.path.join(FLAGS["log.output_dir"], "final_metrics.txt"), "w") as f:
                f.write(f"Accuracy: {final_acc}\nPrecision: {final_pre}\nRecall: {final_rec}\nF1: {final_f1}\nAUC: {final_auc}\n")
                f.write(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}\n")

        except Exception as e:
            print(f"Error computing final metrics: {e}")
