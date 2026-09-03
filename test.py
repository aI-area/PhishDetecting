import pandas as pd
import numpy as np
import pickle
import os
import argparse
import logging
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from feature_extraction import FeatureExtractor
from ngram_processing import NgramProcessor
from model import FocalLossLGBM
from utils import UrlUtils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PhishingTester:
    def __init__(self, artifacts_dir):
        self.artifacts_dir = artifacts_dir
        self.model = None
        self.scaler = None
        self.ngram_processor = None
        self.selected_features_indices = None
        self.extractor = FeatureExtractor()

    def load_artifacts(self):
        """Load all necessary artifacts for inference."""
        try:
            logger.info(f"Loading artifacts from {self.artifacts_dir}...")

            with open(os.path.join(self.artifacts_dir, "model.pkl"), "rb") as f:
                self.model = pickle.load(f)
            
            with open(os.path.join(self.artifacts_dir, "scaler.pkl"), "rb") as f:
                self.scaler = pickle.load(f)

            with open(os.path.join(self.artifacts_dir, "ngram_processor.pkl"), "rb") as f:
                self.ngram_processor = pickle.load(f)

            with open(os.path.join(self.artifacts_dir, "selected_features.pkl"), "rb") as f:
                self.selected_features_indices = pickle.load(f)

            logger.info("All artifacts loaded successfully.")

        except FileNotFoundError as e:
            logger.error(f"Artifact not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise

    def extract_and_transform(self, urls):
        """Extract features for new URLs using saved processors."""
        try:
            
            logger.info("Phase 1: N-gram Extraction (Vectorized)...")
            ngram_matrix = self.ngram_processor.transform(urls)
            
            if hasattr(ngram_matrix, "toarray"):
                ngram_array = ngram_matrix.toarray()
            else:
                ngram_array = ngram_matrix
                
            ngram_cols = self.ngram_processor.get_feature_names()
            ngram_df = pd.DataFrame(ngram_array, columns=ngram_cols, index=urls.index)

            
            logger.info("Phase 2: Handcrafted Extraction (Parallel CPU)...")
            feature_list = Parallel(n_jobs=-1, prefer="threads")(
                delayed(self.extractor.generate_features)(url) 
                for url in tqdm(urls, desc="Handcrafted Features")
            )
            handcrafted_df = pd.DataFrame(feature_list, columns=self.extractor.FEATURE_NAMES, index=urls.index)

            
            logger.info("Phase 3: Merging Feature Sets...")
            final_df = pd.concat([handcrafted_df, ngram_df], axis=1)
            
            return final_df

        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            raise

    def run_test(self, csv_path, output_path=None):
        """Run inference on a new dataset."""
        try:
            self.load_artifacts()

            logger.info(f"Loading test data from {csv_path}...")
            df = pd.read_csv(csv_path, encoding='latin1')
            
 
            if 'url' not in df.columns:
                if 'URL' in df.columns: df.rename(columns={'URL': 'url'}, inplace=True)
                else: raise ValueError("CSV must contain a 'url' column.")
            
            urls = df['url']
            
          
            full_features_df = self.extract_and_transform(urls)

           
            logger.info(f"Selecting {len(self.selected_features_indices)} features...")
            try:
                selected_features_df = full_features_df.iloc[:, self.selected_features_indices]
            except IndexError:
                 logger.error("Feature index mismatch! The new dataset generated different features than training. Ensure N-gram processor matches.")
                 raise

         
            logger.info("Scaling features...")
            features_scaled = self.scaler.transform(selected_features_df.astype(np.float32))

         
            logger.info("Running inference...")
            predictions = self.model.predict(features_scaled)
            probs = self.model.predict_proba(features_scaled)[:, 1]

            
            df['prediction'] = predictions
            df['probability'] = probs
            
            if output_path:
                df.to_csv(output_path, index=False)
                logger.info(f"Results saved to {output_path}")

           
            if 'phishing' in df.columns:
                labels = df['phishing']
                
                
                acc = accuracy_score(labels, predictions)
                prec = precision_score(labels, predictions, zero_division=0)
                rec = recall_score(labels, predictions, zero_division=0)
                f1 = f1_score(labels, predictions, zero_division=0)
                roc = roc_auc_score(labels, probs)
                
                logger.info("-----------------------------")
                logger.info(f"Test Accuracy:  {acc:.4f}")
                logger.info(f"Test Precision: {prec:.4f}")
                logger.info(f"Test Recall:    {rec:.4f}")
                logger.info(f"Test F1 Score:  {f1:.4f}")
                logger.info(f"Test ROC AUC:   {roc:.4f}")
                logger.info("-----------------------------")
                # ------------------------------

            return df

        except Exception as e:
            logger.error(f"Test run failed: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Phishing Model on New Data")
  
    parser.add_argument("csv_path", nargs='?', default="/data/PhishTank.csv", help="Path to CSV dataset")
    parser.add_argument("--artifacts", default="artifacts", help="Path to saved artifacts folder") # change for particular dataset to test on already trained model, e.g, artifacts PhishFusion
    parser.add_argument("--output", default="PhishTank_test_results.csv", help="Path to save output CSV")
    
    args = parser.parse_args()
    
    tester = PhishingTester(args.artifacts)
    tester.run_test(args.csv_path, args.output)

