import numpy as np
import logging
from scipy.special import expit
from lightgbm import LGBMClassifier

class FocalLossLGBM(LGBMClassifier):
    """
    LightGBM with custom focal loss 
    """

    def __init__(
        self,
        gamma=0.30,        
        alpha=0.95,        
        boosting_type='gbdt',
        num_leaves=50,
        max_depth=-1,
        learning_rate=0.15,
        n_estimators=1000,
        min_child_samples=40,
        reg_alpha=0.1,
        reg_lambda=0.3,
        verbosity=-1,
        random_state=42,
        device='cpu',           
        **kwargs
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)
        params = {
            'boosting_type': boosting_type,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'min_child_samples': min_child_samples,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'verbosity': verbosity,
            'random_state': random_state,
            'num_class': 1,
            'device': device
        }
        kwargs.pop('objective', None)
        params.update(kwargs)

        super().__init__(objective=self.custom_obj, **params)

    def custom_obj(self, y_true, y_pred_logits):
        """
       Computes gradient and Hessian for Focal Loss w.r.t. logits (z).
       Correct formulation from Lin et al. (2017): 
         FL(p_t) = -a_t (1 - p_t)^? log(p_t)
       Gradient: dL/dz = -a_t (1 - p_t)^? (y - p)
       Hessian:  d²L/dz² ˜ a_t (1 - p_t)^? p (1 - p)  [diagonal approx]
       """
    
        try:
            # Convert logits to probabilities
            p = expit(y_pred_logits)
            p = np.clip(p, 1e-15, 1 - 1e-15)  
            y_true = y_true.astype(np.float64)
            # Class-specific probability and alpha
            p_t = np.where(y_true == 1, p, 1 - p)
            alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
            # Focal weight: a_t * (1 - p_t)^?
            focal_weight = alpha_t * np.power(1 - p_t, self.gamma)
            # GRADIENT: -focal_weight * (y - p)
            grad = -focal_weight * (y_true - p)
            # GRADIENT: -focal_weight * (y - p)
            hess = focal_weight * p * (1 - p)
            hess = np.clip(hess, 1e-6, None)  
            return grad, hess

        except Exception as e:
            self.logger.error(f"Error in custom objective: {e}")
            return np.zeros_like(y_pred_logits), np.zeros_like(y_pred_logits)

    def predict_proba(self, X, **kwargs):
        try:
            raw_logits = self.booster_.predict(X, raw_score=True, **kwargs)
            prob_class_1 = expit(raw_logits)
            prob_class_0 = 1 - prob_class_1
            return np.vstack((prob_class_0, prob_class_1)).T
        except Exception as e:
            self.logger.error(f"Error in predict_proba: {e}")
            return np.zeros((X.shape[0], 2))

    def predict(self, X, **kwargs):
        try:
            probs = self.predict_proba(X, **kwargs)
            return (probs[:, 1] >= 0.5).astype(int)
        except Exception as e:
            self.logger.error(f"Error in predict: {e}")
            return np.zeros(X.shape[0], dtype=int)