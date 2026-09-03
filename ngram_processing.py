import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
import logging
from constants import RANDOM_SEED

class NgramProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
       
        self.vectorizer = CountVectorizer(
            analyzer='char', 
            ngram_range=(2, 3), 
            min_df=5,  
            dtype=np.float32
        )
        self.selected_ngram_indices = None
        self.is_fitted = False

    def fit_vectorizer(self, urls):
      
        self.logger.info("Fitting N-gram Vectorizer...")
        try:
            self.vectorizer.fit(urls)
            self.is_fitted = True
            vocab_size = len(self.vectorizer.vocabulary_)
            self.logger.info(f"Total n-gram features fitted: {vocab_size}")
        except Exception as e:
            self.logger.error(f"Error fitting vectorizer: {e}")
            raise

    def select_features(self, train_urls, train_labels, val_urls=None, val_labels=None):
        
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before selecting features.")

        self.logger.info("Transforming train URLs to N-gram matrix...")
        X_train = self.vectorizer.transform(train_urls) 
        y_train = train_labels.values

        
        self.logger.info("Applying Variance Threshold...")
        mean = np.array(X_train.mean(axis=0)).flatten()
        sq_mean = np.array(X_train.power(2).mean(axis=0)).flatten()
        var = sq_mean - mean**2
        

        variance_mask = var > 1e-5
        X_train_filtered = X_train[:, variance_mask]
        
        
        original_indices = np.where(variance_mask)[0]
        self.logger.info(f"Reduced n-gram features to {len(original_indices)}")

        
        MAX_SAMPLES_MI = 20000
        if X_train_filtered.shape[0] > MAX_SAMPLES_MI:
            self.logger.info(f"Subsampling {MAX_SAMPLES_MI} samples for MI...")
            np.random.seed(RANDOM_SEED)

            indices = np.random.choice(X_train_filtered.shape[0], MAX_SAMPLES_MI, replace=False)
            X_mi = X_train_filtered[indices]
            y_mi = y_train[indices]
        else:
            X_mi, y_mi = X_train_filtered, y_train

        self.logger.info("Calculating Mutual Information scores...")
        mi_scores = mutual_info_classif(X_mi, y_mi, discrete_features=True, random_state=RANDOM_SEED)

     
        self.logger.info("Training Lasso...")
        lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, random_state=RANDOM_SEED, max_iter=100)
        lasso.fit(X_train_filtered, y_train)
        lasso_weights = np.abs(lasso.coef_[0])

        
        mi_norm = mi_scores / (np.max(mi_scores) + 1e-9)
        lasso_norm = lasso_weights / (np.max(lasso_weights) + 1e-9)
        combined_scores = 0.5 * mi_norm + 0.5 * lasso_norm
        
      
        TOP_K = 5000
        top_k_indices_local = np.argsort(combined_scores)[::-1][:TOP_K]
        
        
        self.selected_ngram_indices = original_indices[top_k_indices_local]
        self.selected_ngram_indices.sort()
        
        self.logger.info(f"Selected {len(self.selected_ngram_indices)} top n-gram features.")

    def transform(self, urls):
        """
        Transforms URLs into the selected N-gram feature matrix.
        """
        if self.selected_ngram_indices is None:
            raise ValueError("Features have not been selected yet.")
            
        try:
            
            full_matrix = self.vectorizer.transform(urls)
            
            
            selected_matrix = full_matrix[:, self.selected_ngram_indices]
            
            return selected_matrix
            
        except Exception as e:
            self.logger.error(f"Transform error: {e}")
            raise

    def get_feature_names(self):
        if self.selected_ngram_indices is None:
            return []
        all_names = np.array(self.vectorizer.get_feature_names_out())
        return all_names[self.selected_ngram_indices].tolist()