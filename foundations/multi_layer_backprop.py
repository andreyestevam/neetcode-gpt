import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        n = len(y_true)
        
        # Forward pass
        z1 = W1 @ x + b1 # x 1D, W1 2D of size (x.length by weight.length) so need to transpose
        a1 = np.maximum(0.0, z1) # ReLU(z1)
        z2 = W2 @ a1 + b2
        y_hat = z2
        L = np.mean((y_hat - y_true) ** 2) # MSE

        # Backwards pass
        dL_dz2 = 2 * (z2 - y_true) / n
        dL_dW2 = np.outer(dL_dz2, a1)
        dL_db2 = dL_dz2

        dL_da1 = dL_dz2 @ W2
        dL_dz1 = dL_da1 * (z1 > 0)
        dL_dW1 = np.outer(dL_dz1, x)
        dL_db1 = dL_dz1

        return {
            'loss': round(float(L), 4),
            'dW1': [[round(float(v), 4) for v in row] for row in dL_dW1],
            'db1': [round(float(v), 4) for v in dL_db1],
            'dW2': [[round(float(v), 4) for v in row] for row in dL_dW2],
            'db2': [round(float(v), 4) for v in dL_db2],
        }
