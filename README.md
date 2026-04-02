# Physics-Informed Ensemble for Legged Robot Fault Diagnosis

This repository provides the official implementation of the paper:

"Physics-Informed Ensemble Learning for Hierarchical Fault Diagnosis in Quadruped Robots", published in Results in Engineering (2026).


## Key Features

- **Physics-informed feature engineering**: Statistical, physics-inspired, and fault-specific descriptors
- **Ensemble learning**: Soft voting of LightGBM, XGBoost, and Random Forest
- **Hierarchical diagnosis**: Joint detection and classification with sub-millisecond latency
- **Interpretable and robust**: Validated on 10,000+ locomotion trials

## Installation

```bash
pip install -r requirements.txt
