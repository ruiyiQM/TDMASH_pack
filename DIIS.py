import numpy as np

class DIIS:
    def __init__(self, subspace_size=8, min_subspace_size=2, regularization=1e-4):
        self.subspace_size = subspace_size
        self.min_subspace_size = min_subspace_size
        self.regularization = regularization
        self.reset()

    def reset(self):
        self.error_vectors = []
        self.param_vectors = []

    def update(self, param, error):
        # Store copies
        self.param_vectors.append(param.copy())
        self.error_vectors.append(error.copy())

        if len(self.param_vectors) > self.subspace_size:
            self.param_vectors.pop(0)
            self.error_vectors.pop(0)

    def extrapolate(self):
        n = len(self.param_vectors)
        if n < self.min_subspace_size:
            return self.param_vectors[-1]

        # Build B matrix
        B = np.zeros((n + 1, n + 1), dtype=np.complex128)
        for i in range(n):
            ei = self.error_vectors[i].ravel()
            for j in range(i, n):
                ej = self.error_vectors[j].ravel()
                val = np.vdot(ei, ej)
                B[i, j] = val
                B[j, i] = np.conj(val)
        
        # Regularization
        for i in range(n): B[i, i] += self.regularization
        
        B[n, :n] = -1.0
        B[:n, n] = -1.0
        B[n, n] = 0.0

        rhs = np.zeros(n + 1, dtype=np.complex128)
        rhs[n] = -1.0

        # Robust solve using Pseudo-Inverse (pinv)
        try:
            inv_B = np.linalg.pinv(B, rcond=1e-12)
            coeffs = (inv_B @ rhs)[:n]
        except np.linalg.LinAlgError:
            return self.param_vectors[-1]

        extrapolated_param = np.zeros_like(self.param_vectors[0])
        for i in range(n):
            extrapolated_param += coeffs[i] * self.param_vectors[i]

        return extrapolated_param