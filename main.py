import pandas as pd
import numpy as np
import logging
import warnings
import argparse
import sys
import os
import pickle
import random
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from tldextract import extract as extract_tld_parts
from lightgbm import LGBMClassifier
from constants import TRAIN_SIZE, VAL_SIZE, TEST_SIZE, RANDOM_SEED
from utils import UrlUtils
from feature_extraction import FeatureExtractor
from ngram_processing import NgramProcessor
from feature_selection import FeatureSelector
from model import FocalLossLGBM


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

sys.setrecursionlimit(2000)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("lightgbm").setLevel(logging.CRITICAL)


class PhishingDetector:
    """Main class for phishing detection pipeline."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.utils = UrlUtils()
        self.extractor = FeatureExtractor()
        self.ngram_processor = NgramProcessor()
        self.feature_selector = FeatureSelector()
        self.scaler = StandardScaler()
        
        
        self.all_feature_names_for_saving = None
        self.model_to_save = None
        self.train_features_selected_to_save = None
        self.test_features_scaled_to_save = None
        self.test_labels_to_save = None
        self.predictions_to_save = None
        self.selected_features_indices_to_save = None

    def load_data(self, file_path="data/PhishFusion.csv"):
         """Load the dataset."""
         try:
             self.logger.info("Loading dataset")
             df = pd.read_csv(file_path, encoding='latin1')

             url_list   = df['url']
             label_list = df['phishing']

             self.logger.info(f"Dataset loaded with {len(url_list)} samples.")
             return url_list, label_list

         except Exception as e:
             self.logger.error(f"Error loading data: {e}")
             return pd.Series([], dtype='object'), pd.Series([], dtype='object')

    def split_data_indices(self, urls, labels):
        """
        Perform Domain-Stratified Split to prevent Data Leakage.
        Returns INDICES for Train, Val, and Test sets.
        """
        try:
            self.logger.info("Extracting domains for Group-Based Splitting...")
            
            # 1. Extract Root Domains for Grouping
            groups = []
            for url in tqdm(urls, desc="Grouping Domains"):
                try:
                    ext = extract_tld_parts(str(url))
                    root_domain = f"{ext.domain}.{ext.suffix}"
                    groups.append(root_domain)
                except:
                    groups.append("unknown")
            
            groups = np.array(groups)
            
            # 2. Outer Split: Train vs Temp (Val + Test)
            gss_outer = GroupShuffleSplit(
                n_splits=1, 
                test_size=(VAL_SIZE + TEST_SIZE), 
                random_state=RANDOM_SEED
            )
            
            train_idx, temp_idx = next(gss_outer.split(urls, labels, groups))
            
            # 3. Inner Split: Temp -> Val vs Test
            temp_groups = groups[temp_idx]
            temp_urls   = urls.iloc[temp_idx]
            temp_labels = labels.iloc[temp_idx]
            
            relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
            
            gss_inner = GroupShuffleSplit(
                n_splits=1, 
                test_size=relative_test_size, 
                random_state=RANDOM_SEED
            )
            
            val_relative_idx, test_relative_idx = next(gss_inner.split(temp_urls, temp_labels, temp_groups))
            
            # Map relative indices back to original indices
            val_idx = temp_idx[val_relative_idx]
            test_idx = temp_idx[test_relative_idx]
            
            self.logger.info(f"Indices Generated - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
            
            # VERIFICATION
            train_domains = set(groups[train_idx])
            test_domains = set(groups[test_idx])
            overlap = train_domains.intersection(test_domains)
            self.logger.info(f"CRITICAL CHECK: Domain Overlap between Train and Test = {len(overlap)}")
            if len(overlap) > 0:
                self.logger.warning(f"Leakage Detected! Overlapping domains: {list(overlap)[:5]}")
            else:
                self.logger.info("SUCCESS: No Data Leakage detected.")
                
            return train_idx, val_idx, test_idx

        except Exception as e:
            self.logger.error(f"Error splitting data: {e}", exc_info=True)
            return None, None, None

    def extract_features(self, urls):
       
        try:
            self.logger.info("Phase 1: Batch N-gram Extraction (Vectorized)...")
            ngram_matrix = self.ngram_processor.transform(urls)
            
            if hasattr(ngram_matrix, "toarray"):
                ngram_array = ngram_matrix.toarray()
            else:
                ngram_array = ngram_matrix
                
            ngram_cols = self.ngram_processor.get_feature_names()
            ngram_df = pd.DataFrame(ngram_array, columns=ngram_cols, index=urls.index)

            self.logger.info("Phase 2: Handcrafted Extraction (Parallel CPU)...")
            feature_list = Parallel(n_jobs=-1, prefer="threads")(
                delayed(self.extractor.generate_features)(url) 
                for url in tqdm(urls, desc="Handcrafted Features")
            )
            
            handcrafted_df = pd.DataFrame(feature_list, columns=self.extractor.FEATURE_NAMES, index=urls.index)
            
            self.logger.info("Phase 3: Merging Feature Sets...")
            final_df = pd.concat([handcrafted_df, ngram_df], axis=1)
            
            self.all_feature_names_for_saving = list(final_df.columns)
            self.logger.info(f"Feature Extraction Complete. Shape: {final_df.shape}")
            return final_df

        except Exception as e:
            self.logger.critical(f"CRITICAL FAILURE in Feature Extraction: {e}", exc_info=True)
            raise e

    def evaluate_hard_negatives(self, test_df, preds):
        """
        Source Bias Check: Evaluates model on 'Hard Negatives' 
        (Benign URLs that structurally mimic Phishing).
        """
        self.logger.info("\n--- Hard Negative Analysis (Source Bias Check) ---")
        try:
            # 1. Isolate Benign Samples in Test Set
            benign_indices = test_df[test_df['label'] == 0].index
            if len(benign_indices) == 0: 
                self.logger.info("No benign samples in test set.")
                return

            benign_subset = test_df.loc[benign_indices].copy()
            
            # 2. Assign Predictions
            if len(preds) == len(test_df):
                mask = (test_df['label'] == 0).values
                benign_subset['pred'] = preds[mask]
            else:
                self.logger.warning("Prediction length mismatch. Skipping hard negative check.")
                return

            # 3. Define Hard Negatives (Length > 100 OR Depth > 5)
            hard_mask = (benign_subset['url'].str.len() > 100) | (benign_subset['url'].str.count('/') > 5)
            hard_negatives = benign_subset[hard_mask]
            
            # 4. Calculate False Positive Rate
            if len(hard_negatives) > 0:
                false_positives = hard_negatives['pred'].sum()
                fpr = false_positives / len(hard_negatives)
                
                self.logger.info(f"Criteria: Length > 100 or Depth > 4")
                self.logger.info(f"Total Hard Benign URLs: {len(hard_negatives)}")
                self.logger.info(f"False Positives:        {false_positives}")
                self.logger.info(f"Hard Negative FPR:      {fpr:.4%}")
                
                with open("results_hard_negatives.txt", "w") as f:
                    f.write(f"Hard Negative FPR: {fpr:.4f}\nCount: {len(hard_negatives)}\n")
            else:
                self.logger.info("No hard negatives found matching criteria.")
        except Exception as e:
            self.logger.error(f"Hard negative analysis failed: {e}")

    def run_pipeline(self):
        """Execute the phishing detection pipeline."""
        try:
            self.logger.info("Starting pipeline")
            urls, labels = self.load_data()
            if urls.empty:
                raise ValueError("No data loaded")

            # 1. Domain-Stratified Split
            train_idx, val_idx, test_idx = self.split_data_indices(urls, labels)
            
            if train_idx is None:
                raise ValueError("Data splitting failed.")

            train_urls = urls.iloc[train_idx]
            train_labels = labels.iloc[train_idx]
            
            val_urls = urls.iloc[val_idx]
            val_labels = labels.iloc[val_idx]
            
            test_urls = urls.iloc[test_idx]
            test_labels = labels.iloc[test_idx]
            
            self.logger.info(f"Test labels distribution:\n{test_labels.value_counts()}")

            # 2. Fit & select n-gram features
            self.logger.info("Processing n-grams...")
            self.ngram_processor.fit_vectorizer(train_urls)
            self.ngram_processor.select_features(
                train_urls,
                train_labels,
                val_urls,
                val_labels
            )

            # 3. Extract features for ALL URLs
            features_dataframe = self.extract_features(urls)

            # 4. Slice Feature Matrix
            self.logger.info("Slicing feature dataframe using grouped indices...")
            
            train_features = features_dataframe.iloc[train_idx].reset_index(drop=True)
            train_labels   = labels.iloc[train_idx].reset_index(drop=True)
            
            val_features   = features_dataframe.iloc[val_idx].reset_index(drop=True)
            val_labels     = labels.iloc[val_idx].reset_index(drop=True)
            
            test_features  = features_dataframe.iloc[test_idx].reset_index(drop=True)
            test_labels    = labels.iloc[test_idx].reset_index(drop=True)
            
            self.test_labels_to_save = test_labels

            # 5. feature selection
            self.logger.info("Performing feature selection...")
            selected_features = self.feature_selector.perform_merged_selection(
                train_features,
                train_labels,
                stability_runs=20,
                sample_fraction=0.905,
                frequency_threshold=0.705,
                regularization_strength=0.191,
                mi_weight=0.318,
                alpha=0.767,
                num_features=1000,
                correlation_threshold=0.956,  
                sample_size=max(20000, len(urls)),
                random_seed=RANDOM_SEED
            )
            
            if len(selected_features) == 0:
                self.logger.warning("No features selected. Using fallback.")
                selected_features = np.arange(train_features.shape[1])

            self.selected_features_indices_to_save = selected_features

            # 6. Subset and scale
            train_features_selected = train_features.iloc[:, selected_features]
            test_features_selected = test_features.iloc[:, selected_features]
            self.train_features_selected_to_save = train_features_selected

            train_features_scaled = self.scaler.fit_transform(train_features_selected.astype(np.float32))
            test_features_scaled = self.scaler.transform(test_features_selected.astype(np.float32))
            self.test_features_scaled_to_save = test_features_scaled
            self.logger.info("Features scaled.")

            # 7. Train Focal-Loss LightGBM
            self.logger.info("Training Focal Loss LightGBM...")
            model = FocalLossLGBM(
                gamma=0.30,
                alpha=0.95,
                boosting_type='gbdt',
                num_leaves=50,
                max_depth=-1,
                learning_rate=0.15,
                n_estimators=500,
                min_child_samples=40,
                reg_alpha=0.1,
                reg_lambda=0.3,
                verbosity=-1,
                random_state=RANDOM_SEED,
                device='cpu',
                gpu_platform_id=0,
                gpu_device_id=0
            )
            model.fit(train_features_scaled, train_labels)
            self.model_to_save = model

            # 8. Evaluate on test set
            self.logger.info("Evaluating model on test set.")
            predictions = model.predict(test_features_scaled)
            self.predictions_to_save = predictions
            prediction_probabilities = model.predict_proba(test_features_scaled)[:, 1]

            metrics = {
                "Accuracy": accuracy_score(test_labels, predictions),
                "Precision": precision_score(test_labels, predictions, zero_division=0),
                "Recall": recall_score(test_labels, predictions, zero_division=0),
                "F1 Score": f1_score(test_labels, predictions, zero_division=0),
                "ROC AUC": roc_auc_score(test_labels, prediction_probabilities)
            }
            for metric, value in metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")

            results_file_path = "results.txt" 
            with open(results_file_path, "w") as f:
                f.write("Test Set Metrics (Domain Stratified):\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
            self.logger.info(f"Metrics saved to {results_file_path}")

            # 8. --- Train Standard LightGBM Baseline 
            self.logger.info("Training Standard LightGBM (Baseline)...")
            baseline_model = LGBMClassifier(
                boosting_type='gbdt',
                num_leaves=50,
                max_depth=-1,
                learning_rate=0.15,
                n_estimators=500,
                min_child_samples=40,
                reg_alpha=0.1,
                reg_lambda=0.3,
                verbosity=-1,
                random_state=RANDOM_SEED,
                device='cpu'
            )
            baseline_model.fit(train_features_scaled, train_labels)

            self.logger.info("Evaluating Standard LightGBM (Baseline) on test set.")
            baseline_predictions = baseline_model.predict(test_features_scaled)
            baseline_prediction_probabilities = baseline_model.predict_proba(test_features_scaled)[:, 1]

            baseline_metrics = {
                "Accuracy": accuracy_score(test_labels, baseline_predictions),
                "Precision": precision_score(test_labels, baseline_predictions, zero_division=0),
                "Recall": recall_score(test_labels, baseline_predictions, zero_division=0),
                "F1 Score": f1_score(test_labels, baseline_predictions, zero_division=0),
                "ROC AUC": roc_auc_score(test_labels, baseline_prediction_probabilities)
            }
            for metric, value in baseline_metrics.items():
                self.logger.info(f"Baseline {metric}: {value:.4f}")
            
            # --- Hard Negative Analysis ---
            # Create a dataframe for analysis using the TEST indices
            test_analysis_df = pd.DataFrame({
                'url': test_urls.reset_index(drop=True),
                'label': test_labels.reset_index(drop=True)
            })
            self.evaluate_hard_negatives(test_analysis_df, predictions)

            # 9. Save artifacts
            self.logger.info("Saving artifacts...")
            output_data_dir = "artifacts"
            os.makedirs(output_data_dir, exist_ok=True)

            artifacts_to_save = {
                "model.pkl": self.model_to_save,
                "train_features_selected.pkl": self.train_features_selected_to_save,
                "test_features_scaled.pkl": self.test_features_scaled_to_save,
                "test_labels.pkl": self.test_labels_to_save,
                "predictions.pkl": self.predictions_to_save,
                "selected_features.pkl": self.selected_features_indices_to_save,
                "all_feature_names.pkl": self.all_feature_names_for_saving,
                "scaler.pkl": self.scaler,
                "ngram_processor.pkl": self.ngram_processor 
            }

            for filename, artifact in artifacts_to_save.items():
                if artifact is None: continue
                path = os.path.join(output_data_dir, filename)
                try:
                    with open(path, "wb") as f:
                        pickle.dump(artifact, f)
                except Exception as e:
                    self.logger.error(f"Failed to save {filename}: {e}")
            self.logger.info("Artifact saving process complete.")

        except ValueError as ve:
            self.logger.error(f"Pipeline execution failed: {ve}", exc_info=True)
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phishing detection pipeline")
    args = parser.parse_args()
    detector = PhishingDetector()
    detector.run_pipeline()
