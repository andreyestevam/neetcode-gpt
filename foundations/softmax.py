import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        results = []
        max_z = np.max(z)
        for logit in z:
            results.append(np.exp(logit - max_z)/np.sum(np.exp(z - max_z)))
        return np.round(np.array(results), 4)