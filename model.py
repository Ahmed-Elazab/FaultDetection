"""
Physics-Informed Ensemble for hierarchical fault diagnosis.
"""

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


class PhysicsInformedEnsemble:
    """
    Hierarchical ensemble for fault detection and classification.
    
    The model uses LightGBM, XGBoost, and RandomForest with soft voting.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.weights = [0.4, 0.35, 0.25]  # [LightGBM, XGBoost, RandomForest]
        self.detectors_ = None
        self.classifiers_ = None
        self.threshold_ = None
        self.global_mean_ = None
        self.global_std_ = None
    
    def _get_base_models(self):
        """Return base models with consistent hyperparameters."""
        return [
            LGBMClassifier(n_estimators=200, n_jobs=-1, verbose=-1, random_state=self.random_state),
            XGBClassifier(n_estimators=200, n_jobs=-1, verbosity=0, random_state=self.random_state),
            RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=self.random_state)
        ]
    
    def fit(self, X_raw, y_has_fault, y_fault_type):
        """
        Train the ensemble on raw sensor data.
        
        Parameters
        ----------
        X_raw : np.ndarray
            Raw sensor data [N, 300, 107].
        y_has_fault : np.ndarray
            Binary fault indicator [N,].
        y_fault_type : np.ndarray
            Fault type labels (0-6 for 7 faults) [N,].
        """
        # Extract features
        X_features, self.global_mean_, self.global_std_ = extract_features(
            X_raw, global_mean=None, global_std=None
        )
        
        # Train detectors
        self.detectors_ = self._get_base_models()
        detector_probs = []
        for i, detector in enumerate(self.detectors_):
            detector.fit(X_features, y_has_fault)
            proba = detector.predict_proba(X_features)[:, 1]
            detector_probs.append(proba)
        
        # Optimize detection threshold
        ensemble_proba_detect = np.average(detector_probs, weights=self.weights, axis=0)
        precision, recall, thresholds = self._precision_recall_curve(y_has_fault, ensemble_proba_detect)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        self.threshold_ = thresholds[np.argmax(f1_scores)]
        
        # Train classifiers (on true faults only)
        mask_train_fault = (y_has_fault == 1)
        self.classifiers_ = self._get_base_models()
        for i, classifier in enumerate(self.classifiers_):
            if mask_train_fault.sum() > 0:
                sample_weight = compute_sample_weight('balanced', y_fault_type[mask_train_fault])
                classifier.fit(
                    X_features[mask_train_fault], 
                    y_fault_type[mask_train_fault], 
                    sample_weight=sample_weight
                )
    
    def predict(self, X_raw):
        """
        Predict fault presence and type.
        
        Returns
        -------
        y_has_fault_pred : np.ndarray
            Binary predictions [N,].
        y_fault_type_pred : np.ndarray
            Fault type predictions (0 = no fault, 1-7 = fault types) [N,].
        """
        X_features, _, _ = extract_features(
            X_raw, global_mean=self.global_mean_, global_std=self.global_std_
        )
        
        # Detection
        detector_probs = []
        for detector in self.detectors_:
            proba = detector.predict_proba(X_features)[:, 1]
            detector_probs.append(proba)
        ensemble_proba_detect = np.average(detector_probs, weights=self.weights, axis=0)
        y_has_fault_pred = (ensemble_proba_detect >= self.threshold_).astype(int)
        
        # Classification
        y_fault_type_pred = np.zeros_like(y_has_fault_pred)
        if np.any(y_has_fault_pred == 1):
            classifier_probs = []
            for classifier in self.classifiers_:
                proba = classifier.predict_proba(X_features)
                classifier_probs.append(proba)
            ensemble_proba_classify = np.average(classifier_probs, weights=self.weights, axis=0)
            y_fault_type_pred[y_has_fault_pred == 1] = np.argmax(
                ensemble_proba_classify[y_has_fault_pred == 1], axis=1
            ) + 1  # Convert 0-6 → 1-7
        
        return y_has_fault_pred, y_fault_type_pred
    
    @staticmethod
    def _precision_recall_curve(y_true, y_proba):
        """Simple PR curve implementation."""
        from sklearn.metrics import precision_recall_curve
        return precision_recall_curve(y_true, y_proba)