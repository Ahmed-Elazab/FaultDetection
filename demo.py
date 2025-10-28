"""
Example usage of the Physics-Informed Ensemble.
"""

import numpy as np
from src.model import PhysicsInformedEnsemble

# Simulate raw sensor data: [N, 300, 107]
np.random.seed(42)
X_raw = np.random.randn(100, 300, 107).astype(np.float32)

# Simulate labels
y_has_fault = np.random.choice([0, 1], size=100, p=[0.5, 0.5])
y_fault_type = np.random.choice(7, size=100)  # 0-6 for 7 faults
y_fault_type[y_has_fault == 0] = 0  # Normal samples have type 0

# Train model
model = PhysicsInformedEnsemble(random_state=42)
model.fit(X_raw, y_has_fault, y_fault_type)

# Predict
y_hf_pred, y_ft_pred = model.predict(X_raw)

print("Prediction shapes:", y_hf_pred.shape, y_ft_pred.shape)
print("Sample predictions (first 10):")
for i in range(10):
    print(f"  Trial {i}: has_fault={y_hf_pred[i]}, fault_type={y_ft_pred[i]}")