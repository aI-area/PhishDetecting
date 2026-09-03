
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ["KMP_WARNINGS"] = "0"         
os.environ["KMP_AFFINITY"] = "disabled"   

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import   *
import os
import pickle
import time
from tqdm import tqdm
import argparse
import numpy as np
import tensorflow as tf
from tflearn.data_utils import to_categorical, pad_sequences
import csv
import tempfile
from pathlib import Path

# ----------------------------- Argparser -----------------------------
parser = argparse.ArgumentParser(description="Test URLNet model")

# data args
default_max_len_words = 200
parser.add_argument('--data.max_len_words', type=int, default=default_max_len_words, metavar="MLW",
    help="maximum length of url in words (default: {})".format(default_max_len_words))
default_max_len_chars = 200
parser.add_argument('--data.max_len_chars', type=int, default=default_max_len_chars, metavar="MLC",
    help="maximum length of url in characters (default: {})".format(default_max_len_chars))
default_max_len_subwords = 20
parser.add_argument('--data.max_len_subwords', type=int, default=default_max_len_subwords, metavar="MLSW",
    help="maximum length of word in subwords/characters (default: {})".format(default_max_len_subwords))
parser.add_argument('--data.data_dir', type=str, default='train_10000.txt', metavar="DATADIR",
    help="path to data file (.txt in URLNet format, or .csv with columns url, phishing)")
default_delimit_mode = 1
parser.add_argument("--data.delimit_mode", type=int, default=default_delimit_mode, metavar="DLMODE",
    help="0: delimit by special chars, 1: special chars + each char as a word (default: {})".format(default_delimit_mode))

# dictionaries (produced by train.py)
parser.add_argument('--data.subword_dict_dir', type=str, default="runs/10000/subwords_dict.p", metavar="SUBWORD_DICT",
    help="path to the subword (ngram) dictionary pickle")
parser.add_argument('--data.word_dict_dir', type=str, default="runs/10000/words_dict.p", metavar="WORD_DICT",
    help="path to the word dictionary pickle")
parser.add_argument('--data.char_dict_dir', type=str, default="runs/10000/chars_dict.p", metavar="CHAR_DICT",
    help="path to the character dictionary pickle")

# model args
default_emb_dim = 32
parser.add_argument('--model.emb_dim', type=int, default=default_emb_dim, metavar="EMBDIM",
    help="embedding dimension size (default: {})".format(default_emb_dim))
default_emb_mode = 1
parser.add_argument('--model.emb_mode', type=int, default=default_emb_mode, metavar="EMBMODE",
    help="1: charCNN, 2: wordCNN, 3: char + wordCNN, 4: char-level wordCNN, 5: char + char-level wordCNN")

# test args
default_batch_size = 128
parser.add_argument('--test.batch_size', type=int, default=default_batch_size, metavar="BATCHSIZE",
    help="size of each test batch (default: {})".format(default_batch_size))

# log / checkpoints
parser.add_argument('--log.output_dir', type=str, default="runs/10000/", metavar="OUTPUTDIR", 
    help="directory to save the test results")
parser.add_argument('--log.checkpoint_dir', type=str, default="runs/10000/checkpoints/", metavar="CHECKPOINTDIR",  
    help="directory containing the learned model checkpoints")

# CSV options (only used if --data.data_dir ends with .csv)
parser.add_argument('--csv.url_col', type=str, default='url', help="CSV column name for URL")
parser.add_argument('--csv.label_col', type=str, default='phishing', help="CSV column name for label (1/0)")

FLAGS = vars(parser.parse_args())
for key, val in FLAGS.items():
    print("{}={}".format(key, val))

# ---------------- CSV auto-conversion ----------------

def _find_col_case_insensitive(fieldnames, target):
    m = {name.lower(): name for name in fieldnames}
    if target.lower() not in m:
        raise ValueError("CSV is missing required column '{}'. Found: {}".format(target, fieldnames))
    return m[target.lower()]

def _csv_to_urlnet_txt(input_csv, url_col_name="url", label_col_name="phishing"):
    input_csv = Path(input_csv)
    out_path = Path(tempfile.gettempdir()) / "{}_urlnet.txt".format(input_csv.stem)
    
    # Using errors="ignore" to skip bad bytes in URLs
    with input_csv.open("r", newline="", encoding="utf-8", errors="ignore") as f_in, \
         out_path.open("w", newline="\n", encoding="utf-8") as f_out:
         
        reader = csv.DictReader(f_in)
        url_col = _find_col_case_insensitive(reader.fieldnames, url_col_name)
        lab_col = _find_col_case_insensitive(reader.fieldnames, label_col_name)
        n_written = 0
        for row in reader:
            url = (row[url_col] or "").strip()
            lab_raw = row[lab_col]
            try:
                lab = int(lab_raw)
            except Exception:
                continue
            mapped = "+1" if lab == 1 else "-1"
            if url:
                f_out.write("{}\t{}\n".format(mapped, url))
                n_written += 1
    print("[convert] Wrote {} lines to {}".format(n_written, out_path))
    return str(out_path)
    
data_path = FLAGS["data.data_dir"]
if str(data_path).lower().endswith(".csv"):
    print("[convert] Detected CSV: {} -> converting to URLNet format...".format(data_path))
    FLAGS["data.data_dir"] = _csv_to_urlnet_txt(
        data_path, url_col_name=FLAGS["csv.url_col"], label_col_name=FLAGS["csv.label_col"]
    )
    print("[convert] Using converted file: {}".format(FLAGS["data.data_dir"]))

# ---------------- Load & preprocess data ----------------
# NOTE: This uses read_data from utils.py (must be the TF1 version)
urls, labels = read_data(FLAGS["data.data_dir"])

x, word_reverse_dict = get_word_vocab(urls, FLAGS["data.max_len_words"])
word_x = get_words(x, word_reverse_dict, FLAGS["data.delimit_mode"], urls)

ngram_dict = pickle.load(open(FLAGS["data.subword_dict_dir"], "rb"))
print("Size of subword vocabulary (train): {}".format(len(ngram_dict)))
word_dict = pickle.load(open(FLAGS["data.word_dict_dir"], "rb"))
print("Size of word vocabulary (train): {}".format(len(word_dict)))

ngramed_id_x, worded_id_x = ngram_id_x_from_dict(
    word_x, FLAGS["data.max_len_subwords"], ngram_dict, word_dict
)

chars_dict = pickle.load(open(FLAGS["data.char_dict_dir"], "rb"))
chared_id_x = char_id_x(urls, chars_dict, FLAGS["data.max_len_chars"])

print("Number of testing urls: {}".format(len(labels)))

# ---------------- Evaluation ----------------
def test_step(x, emb_mode):
    p = 1.0
    if emb_mode == 1:
        feed_dict = {input_x_char_seq: x[0], dropout_keep_prob: p}
    elif emb_mode == 2:
        feed_dict = {input_x_word: x[0], dropout_keep_prob: p}
    elif emb_mode == 3:
        feed_dict = {input_x_char_seq: x[0], input_x_word: x[1], dropout_keep_prob: p}
    elif emb_mode == 4:
        feed_dict = {input_x_word: x[0], input_x_char: x[1], input_x_char_pad_idx: x[2], dropout_keep_prob: p}
    elif emb_mode == 5:
        feed_dict = {input_x_char_seq: x[0], input_x_word: x[1], input_x_char: x[2], input_x_char_pad_idx: x[3],
                     dropout_keep_prob: p}
    preds, s = sess.run([predictions, scores], feed_dict)
    return preds, s

checkpoint_file = tf.train.latest_checkpoint(FLAGS["log.checkpoint_dir"])
if checkpoint_file is None:
    raise RuntimeError("No checkpoint found in {}".format(FLAGS["log.checkpoint_dir"]))

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

      
        if FLAGS["model.emb_mode"] in [1, 3, 5]:
            input_x_char_seq = graph.get_operation_by_name("input_x_char_seq").outputs[0]
        if FLAGS["model.emb_mode"] in [2, 3, 4, 5]:
            input_x_word = graph.get_operation_by_name("input_x_word").outputs[0]
        if FLAGS["model.emb_mode"] in [4, 5]:
            input_x_char = graph.get_operation_by_name("input_x_char").outputs[0]
            input_x_char_pad_idx = graph.get_operation_by_name("input_x_char_pad_idx").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        scores = graph.get_operation_by_name("output/scores").outputs[0]

    
        if FLAGS["model.emb_mode"] == 1:
            batches = batch_iter(list(chared_id_x), FLAGS["test.batch_size"], 1, shuffle=False)
        elif FLAGS["model.emb_mode"] == 2:
            batches = batch_iter(list(worded_id_x), FLAGS["test.batch_size"], 1, shuffle=False)
        elif FLAGS["model.emb_mode"] == 3:
            batches = batch_iter(list(zip(chared_id_x, worded_id_x)), FLAGS["test.batch_size"], 1, shuffle=False)
        elif FLAGS["model.emb_mode"] == 4:
            batches = batch_iter(list(zip(ngramed_id_x, worded_id_x)), FLAGS["test.batch_size"], 1, shuffle=False)
        elif FLAGS["model.emb_mode"] == 5:
            batches = batch_iter(list(zip(ngramed_id_x, worded_id_x, chared_id_x)), FLAGS["test.batch_size"], 1, shuffle=False)

        all_predictions = np.array([], dtype=np.int64) 
        all_scores = []

        nb_batches = int(len(labels) / FLAGS["test.batch_size"])
        if len(labels) % FLAGS["test.batch_size"] != 0:
            nb_batches += 1
        print("Number of batches in total: {}".format(nb_batches))

        it = tqdm(range(nb_batches),
                  desc="emb_mode {} delimit_mode {} test_size {}".format(
                      FLAGS["model.emb_mode"], FLAGS["data.delimit_mode"], len(labels)),
                  ncols=0)

        for idx in it:
            batch = next(batches)

            if FLAGS["model.emb_mode"] == 1:
                x_char_seq = batch
            elif FLAGS["model.emb_mode"] == 2:
                x_word = batch
            elif FLAGS["model.emb_mode"] == 3:
                x_char_seq, x_word = zip(*batch)
            elif FLAGS["model.emb_mode"] == 4:
                x_char, x_word = zip(*batch)
            elif FLAGS["model.emb_mode"] == 5:
                x_char, x_word, x_char_seq = zip(*batch)

            x_batch = []
            if FLAGS["model.emb_mode"] in [1, 3, 5]:
                x_char_seq = pad_seq_in_word(x_char_seq, FLAGS["data.max_len_chars"])
                x_batch.append(x_char_seq)
            if FLAGS["model.emb_mode"] in [2, 3, 4, 5]:
                x_word = pad_seq_in_word(x_word, FLAGS["data.max_len_words"])
                x_batch.append(x_word)
            if FLAGS["model.emb_mode"] in [4, 5]:
                x_char, x_char_pad_idx = pad_seq(x_char, FLAGS["data.max_len_words"],
                                                 FLAGS["data.max_len_subwords"], FLAGS["model.emb_dim"])
                x_batch.extend([x_char, x_char_pad_idx])

            batch_predictions, batch_scores = test_step(x_batch, FLAGS["model.emb_mode"])
            batch_predictions = np.squeeze(np.asarray(batch_predictions)).astype(np.int64)
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            all_scores.extend(batch_scores)

            it.set_postfix()

# ---------------- Metrics + Save results ----------------

def _binary_metrics(y_true, y_pred, y_score_pos):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score_pos = np.asarray(y_score_pos, dtype=float)

    n = y_true.size
    acc = float((y_true == y_pred).sum()) / float(n) if n else float('nan')

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float('nan')

    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float('nan')

   
    pos = (y_true == 1).sum()
    neg = (y_true == 0).sum()
    if pos == 0 or neg == 0:
        roc_auc = float('nan')
    else:
        order = np.argsort(-y_score_pos, kind='mergesort')  # stable, desc
        y_sorted = y_true[order]
        tps = np.cumsum(y_sorted == 1)
        fps = np.cumsum(y_sorted == 0)
        tpr = tps / pos
        fpr = fps / neg
        roc_auc = float(np.trapz(tpr, fpr))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "tp": tp, "fp": fp, "fn": fn, "pos": int(pos), "neg": int(neg), "n": int(n)
    }


probs_pos = []
for s in all_scores:
    
    try:
        p1 = softmax(s)[1]
    except Exception:
        
        s = np.atleast_1d(s)
        if s.size == 1:
            p1 = float(s[0])
        else:
            e = np.exp(s - np.max(s))
            p1 = float((e / e.sum())[-1])
    probs_pos.append(p1)
probs_pos = np.asarray(probs_pos, dtype=float)


if labels is not None and len(labels) == len(all_predictions):
    m = _binary_metrics(labels, all_predictions, probs_pos)
    print("\n=== Evaluation Metrics ===")
    print("Samples: {n} (pos={pos}, neg={neg})  TP={tp} FP={fp} FN={fn}".format(**m))
    print("Accuracy : {:.6f}".format(m["accuracy"]) if np.isfinite(m["accuracy"]) else "Accuracy : N/A")
    print("Precision: {:.6f}".format(m["precision"]) if np.isfinite(m["precision"]) else "Precision: N/A")
    print("Recall   : {:.6f}".format(m["recall"])    if np.isfinite(m["recall"])    else "Recall   : N/A")
    print("F1-Score : {:.6f}".format(m["f1"])        if np.isfinite(m["f1"])        else "F1-Score : N/A")
    print("ROC AUC  : {:.6f}".format(m["roc_auc"])   if np.isfinite(m["roc_auc"])   else "ROC AUC  : N/A")
else:
    print("Labels missing or length mismatch; skipping metrics.")


os.makedirs(FLAGS["log.output_dir"], exist_ok=True)
results_path = os.path.join(FLAGS["log.output_dir"], "test_results.csv")
save_test_result(labels, all_predictions, all_scores, results_path)
print("Saved test results to {}".format(results_path))


