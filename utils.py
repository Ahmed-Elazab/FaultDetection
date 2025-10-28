"""Utility functions."""

import numpy as np

def compute_joint_accuracy(y_has_fault_true, y_has_fault_pred, y_fault_type_true, y_fault_type_pred):
    """Compute hierarchical joint accuracy."""
    correct_detection = (y_has_fault_pred == y_has_fault_true)
    correct_classification = np.zeros_like(correct_detection)
    fault_mask = (y_has_fault_true == 1)
    if np.any(fault_mask):
        correct_classification[fault_mask] = (y_fault_type_pred[fault_mask] == y_fault_type_true[fault_mask])
    joint_correct = correct_detection & ((y_has_fault_true == 0) | (correct_classification & (y_has_fault_true == 1)))
    return np.mean(joint_correct)