# feature_selection.py
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

def run_tabnet_feature_selection(X_train, y_train, feature_names):
    """
    Trains TabNet to compute feature importance.
    Parameters:
    - n_d (Decision steps): 64 [cite: 179]
    - n_a (Attention steps): 64 [cite: 181]
    - Optimizer: Adam [cite: 182]
    - Learning Rate: Exponential Decay with gamma 0.9 [cite: 183-186]
    - Epochs: 10 [cite: 378]
    """
    print("\n--- Starting TabNet Feature Selection ---")
    
    # Initialize TabNet with paper specifications
    clf = TabNetClassifier(
        n_d=64,
        n_a=64,
        n_steps=5, # Default is usually 3-10, paper implies multiple decision steps
        gamma=1.5, # Relaxation parameter, default is 1.3-1.5
        n_independent=2,
        n_shared=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 1, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax', # or 'sparsemax'
        verbose=1
    )
    
    # Train TabNet
    # "The TabNet deep neural network model is trained across 10 epochs" [cite: 378]
    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        eval_name=['train'],
        eval_metric=['auc'],
        max_epochs=10,
        patience=10,
        batch_size=1024, 
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    # Compute Feature Importance
    importances = clf.feature_importances_
    
    # Select Top 50 Features [cite: 381]
    top_n = 50
    # Sort descending
    indices = np.argsort(importances)[::-1][:top_n]
    selected_feats = [feature_names[i] for i in indices]
    
    print(f"\nTabNet Identified Top {len(selected_feats)} Features.")
    
    return indices, selected_feats