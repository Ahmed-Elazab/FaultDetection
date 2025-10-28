# Physics-Informed Ensemble for Legged Robot Fault Diagnosis

This repository implements the **Physics-Informed Ensemble** method for hierarchical fault detection and classification in quadrupedal robots, as described in our paper.

## Key Features

- **Physics-informed feature engineering**: Statistical, physics-inspired, and fault-specific descriptors
- **Ensemble learning**: Soft voting of LightGBM, XGBoost, and Random Forest
- **Hierarchical diagnosis**: Joint detection and classification with sub-millisecond latency
- **Interpretable and robust**: Validated on 10,000+ locomotion trials

## Installation

```bash
pip install -r requirements.txt