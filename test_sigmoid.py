"""Smoke test for sigmoid. Run with: python test_sigmoid.py"""

from logreg_scratch import sigmoid
import numpy as np

result = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
print(f"sigmoid([-1000, 0, 1000]) = {result}")
print(f"Expected:                  [0., 0.5, 1.]")

if np.allclose(result, [0.0, 0.5, 1.0], atol=1e-6):
    print("PASS — sigmoid works.")
else:
    print("FAIL — output does not match expected values.")
