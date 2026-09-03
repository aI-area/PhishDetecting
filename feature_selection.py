import numpy as np
import logging
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from tqdm import tqdm
from joblib import Parallel, delayed
from constants import RANDOM_SEED

class FeatureSelector:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def perform_stability_selection(self, features_np, labels_np, sample_fraction, stability_runs, regularization_strength, rng):
        """Perform stability selection in PARALLEL."""
        try:
            if features_np.size == 0: raise ValueError("Empty features")
            
            def _single_run(seed):
                local_rng = np.random.RandomState(seed)
                
                indices = local_rng.choice(
                    features_np.shape[0],
                    size=int(sample_fraction * features_np.shape[0]),
                    replace=False
                )
                model = LogisticRegression(
                    penalty='l1', solver='liblinear', C=regularization_strength,
                    random_state=local_rng.randint(0, 5000), n_jobs=1
                )
                model.fit(features_np[indices], labels_np[indices])
                
                selected = np.abs(model.coef_) > 1e-5
                return selected.ravel().astype(int)

            seeds = [rng.randint(0, 10000) for _ in range(stability_runs)]
            
            
            results = Parallel(n_jobs=-1)(
                delayed(_single_run)(s) for s in tqdm(seeds, desc="Stability Selection")
            )
            
            frequency_counts = np.sum(results, axis=0)
            return frequency_counts / stability_runs

        except Exception as e:
            self.logger.error(f"Error in stability: {e}")
            return np.zeros(features_np.shape[1])

    def perform_hybrid_selection(self, features_np, labels_np, mi_weight, n_estimators=200):
        
        try:
            self.logger.info("Starting Hybrid Selection...")

            
            n_samples = features_np.shape[0]
            if n_samples > 20000:
                self.logger.info(f"Subsampling 20,000 samples (from {n_samples}) for fast MI calculation...")
                np.random.seed(RANDOM_SEED)
                indices = np.random.choice(n_samples, 20000, replace=False)
                mi_features = features_np[indices]
                mi_labels = labels_np[indices]
            else:
                mi_features = features_np
                mi_labels = labels_np

            self.logger.info("Calculating Mutual Information (MI) scores...")
            mi_scores = mutual_info_classif(mi_features, mi_labels, random_state=RANDOM_SEED)
            self.logger.info("MI scores calculated.")

            
            self.logger.info("Training lightGBM for feature importance...")
            
            try:
                import torch
                has_gpu = torch.cuda.is_available()
            except ImportError:
                has_gpu = False
            try: 
                import torch
                has_gpu = torch.cuda.is_available()
            except ImportError:
                has_gpu = False
            tree_method = "hist" 
            device = "cpu"
            
            self.logger.info(f"Using lightGBM device: {device}")
          
            model = LGBMClassifier(
                n_estimators=n_estimators,
                random_state=RANDOM_SEED,
                importance_type='gain',  
                n_jobs=-1,
                verbose=-1
            )
            
            model.fit(features_np, labels_np)
            xgb_scores = model.feature_importances_
            self.logger.info("XGBoost training complete.")
            
            
            mi_norm = mi_scores / (mi_scores.max() if mi_scores.max() != 0 else 1)
            xgb_norm = xgb_scores / (xgb_scores.max() if xgb_scores.max() != 0 else 1)
            
            return mi_weight * mi_norm + (1 - mi_weight) * xgb_norm

        except Exception as e:
            self.logger.error(f"Error in hybrid: {e}")
            return np.zeros(features_np.shape[1])

    def perform_merged_selection(self, features, labels, stability_runs=20, sample_fraction=0.905,
                                 frequency_threshold=0.705, regularization_strength=0.191,
                                 mi_weight=0.318, alpha=0.767, num_features=1000,
                                 correlation_threshold=0.950, sample_size=20000,
                                 random_seed=RANDOM_SEED, n_estimators=200):

        try:
            features_np = features.values.astype(np.float32)
            labels_np = labels.values
            rng = np.random.RandomState(random_seed)

         
            stab_freq = self.perform_stability_selection(
                features_np, labels_np, sample_fraction, stability_runs, regularization_strength, rng
            )

            
            hybrid_scores = self.perform_hybrid_selection(features_np, labels_np, mi_weight, n_estimators)

           
            merged = alpha * stab_freq + (1 - alpha) * hybrid_scores
            
          
            candidates = np.where(stab_freq >= frequency_threshold)[0]
            if candidates.size == 0:
                self.logger.warning("No stable features found. Using top sorted features instead.")
                candidates = np.arange(len(merged))
            
            
            sorted_indices = candidates[np.argsort(merged[candidates])[::-1]]

     
            self.logger.info("Starting Optimized Greedy Redundancy Pruning...")
            
    
            limit = 5000
            if len(sorted_indices) > limit:
                sorted_indices = sorted_indices[:limit]
            
            candidate_data = features_np[:, sorted_indices]
            
          
            self.logger.info("Computing Correlation Matrix...")
           
            corr_matrix = np.abs(np.corrcoef(candidate_data, rowvar=False))
            np.nan_to_num(corr_matrix, copy=False)
            
            selected = []
            final_indices = []
            is_redundant = np.zeros(len(sorted_indices), dtype=bool)

            self.logger.info("Pruning redundant features...")
            for i in range(len(sorted_indices)):
                if is_redundant[i]: continue
                
                selected.append(i)
                final_indices.append(sorted_indices[i])
                
                if len(final_indices) >= num_features: break
                
             
             
                mask = corr_matrix[i] > correlation_threshold
                mask[:i+1] = False 
                is_redundant |= mask

            self.logger.info(f"Feature Selection Complete. Selected {len(final_indices)} features.")
            return np.array(final_indices, dtype=int)

        except Exception as e:
            self.logger.error(f"Error merged selection: {e}", exc_info=True)
            return np.array([], dtype=int)
