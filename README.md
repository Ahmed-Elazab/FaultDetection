# Physics-Informed Ensemble for Legged Robot Fault Diagnosis

This repository provides the official implementation of the paper:

"Physics-Informed Ensemble Learning for Hierarchical Fault Diagnosis in Quadruped Robots", published in Results in Engineering (2026).

## 📌 Overview
We propose a lightweight, interpretable, and highly efficient ensemble framework that combines domain-guided feature engineering with classical machine learning to solve two tasks simultaneously:

- **Fault detection**: Is the robot operating normally or under fault?
- **Fault classification**: Which of the 7 distinct fault types is occurring?

Unlike black-box deep learning models, our approach is transparent, robust, and deployable, making it ideal for real-world legged robotics.

## 🧠 Framework Architecture
<div align="center">
<img src="https://github.com/user-attachments/assets/cd5bf7f9-79ef-4586-bd3c-458e671a377a" width="90%" alt="Framework Diagram">
</div>

The pipeline consists of three stages:

1. **Physics-Informed Feature Extraction**: 652-D vector from raw sensor streams using statistical, gait-aware, and dynamics-based descriptors.
2. **Ensemble Inference**: Soft-voting fusion of LightGBM, XGBoost, and Random Forest.
3. **Hierarchical Decision**: Joint detection + classification with calibrated confidence.

## ⚙️ Algorithm Workflow
<div align="center">
<img src="https://github.com/user-attachments/assets/4d179b63-e0de-4aeb-b498-b65ef269d0cd" width="70%" alt="Algorithm Flowchart">
</div>


## ✨ Key Features

- **Physics-informed feature engineering**: Statistical, physics-inspired, and fault-specific descriptors
- **Ensemble learning**: Soft voting of LightGBM, XGBoost, and Random Forest
- **Hierarchical diagnosis**: Joint detection and classification with sub-millisecond latency
- **Interpretable and robust**: Validated on 10,000+ locomotion trials

## Installation

```bash
pip install -r requirements.txt
