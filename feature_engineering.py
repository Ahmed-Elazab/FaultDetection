"""
Physics-informed feature engineering for legged robot fault diagnosis.
"""

import numpy as np

# Sensor channel indices (adjust if your data schema differs)
CONTACT_FORCE_IDX = [96, 97, 98, 99]      # FL, FR, RL, RR
JOINT_TORQUE_START = 24
JOINT_TORQUE_END = 48
IMU_ACC_Z_IDX = 90


def compute_statistical_features(window):
    """Compute 6 statistical descriptors per channel."""
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    max_val = np.max(window, axis=0)
    min_val = np.min(window, axis=0)
    
    skew = np.zeros(window.shape[1])
    kurt = np.zeros(window.shape[1])
    for i in range(window.shape[1]):
        if std[i] > 1e-8:
            normalized = (window[:, i] - mean[i]) / std[i]
            skew[i] = np.mean(normalized ** 3)
            kurt[i] = np.mean(normalized ** 4) - 3
    return np.concatenate([mean, std, max_val, min_val, skew, kurt])


def compute_physics_features(trial):
    """Compute physics-inspired features from a full trial."""
    features = []
    contact_forces = trial[:, CONTACT_FORCE_IDX]
    torques = trial[:, JOINT_TORQUE_START:JOINT_TORQUE_END]
    imu_z = trial[:, IMU_ACC_Z_IDX]
    
    # Gait symmetry (front legs)
    fl_torques = torques[:, 0:3]
    fr_torques = torques[:, 3:6]
    sym_fl_fr = np.mean(np.abs(fl_torques - fr_torques))
    features.append(sym_fl_fr)
    
    # Center of Pressure
    F_fl, F_fr, F_rl, F_rr = contact_forces[:, 0], contact_forces[:, 1], contact_forces[:, 2], contact_forces[:, 3]
    total_force = F_fl + F_fr + F_rl + F_rr
    total_force = np.where(total_force == 0, 1e-8, total_force)
    cop_x = (-F_fl + F_fr - F_rl + F_rr) / total_force
    cop_y = (F_fl + F_fr - F_rl - F_rr) / total_force
    features.extend([np.mean(cop_x), np.mean(cop_y)])
    
    # Energy proxy
    energy = np.mean(np.sum(torques ** 2, axis=1))
    features.append(energy)
    
    return np.array(features)


def compute_fault_specific_features(trial):
    """Compute fault-specific descriptors."""
    features = []
    
    # Motor saturation (RF leg: joints 3-5 → torque indices 27-29)
    rf_torques = trial[:, 27:30]
    sat_rf = np.max(np.abs(rf_torques))
    features.append(sat_rf)
    
    # IMU bias
    imu_z = trial[:, IMU_ACC_Z_IDX]
    bias_imu = np.mean(np.abs(imu_z + 9.81))
    features.append(bias_imu)
    
    # Leg slip (RF: contact force index 97, joint velocities 15-17)
    rf_contact = trial[:, 97]
    rf_joint_vel = trial[:, 15:18]
    slip_metric = np.mean(np.sum(np.abs(rf_joint_vel), axis=1) / (rf_contact + 1e-8))
    features.append(slip_metric)
    
    return np.array(features)


def extract_features(X_raw, global_mean=None, global_std=None):
    """
    Extract 652-D feature vectors from raw sensor time series.
    
    Parameters
    ----------
    X_raw : np.ndarray
        Raw sensor data of shape [N, 300, 107].
    global_mean : np.ndarray, optional
        Global mean for normalization (computed if not provided).
    global_std : np.ndarray, optional
        Global std for normalization (computed if not provided).
    
    Returns
    -------
    X_features : np.ndarray
        Feature matrix of shape [N, 652].
    global_mean : np.ndarray
        Computed global mean.
    global_std : np.ndarray
        Computed global std.
    """
    N, T, D = X_raw.shape
    WINDOW_SIZE = 10
    STRIDE = 5
    
    # Normalize
    if global_mean is None:
        global_mean = np.mean(X_raw.reshape(-1, D), axis=0)
    if global_std is None:
        global_std = np.std(X_raw.reshape(-1, D), axis=0)
        global_std = np.where(global_std == 0, 1.0, global_std)
    X_norm = (X_raw - global_mean) / global_std
    
    # Extract features
    X_features = []
    for i in range(N):
        trial_stats = []
        for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
            window = X_norm[i, start:start + WINDOW_SIZE, :]
            stat_feat = compute_statistical_features(window)
            trial_stats.append(stat_feat)
        
        avg_stat = np.mean(trial_stats, axis=0)
        physics_feat = compute_physics_features(X_norm[i])
        fault_feat = compute_fault_specific_features(X_norm[i])
        trial_features = np.concatenate([avg_stat, physics_feat, fault_feat])
        X_features.append(trial_features)
    
    return np.array(X_features), global_mean, global_std