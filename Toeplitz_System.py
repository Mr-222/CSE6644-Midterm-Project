import numpy as np
import scipy


def generate_power_decay_system(n: int, p: np.float64) -> np.ndarray:
    """
    A_n = [
        a_0, a_{-1}, ..., a_{2 - n}, a_{1 - n},
        a_1, a_0, a{-1}, ..., a_{2 - n},
        ...
        a_{n-1}, a_{n-2}, ..., a_1, a_0
    ]
    A:SPD
    a_k = | k + 1 |^(-p) for lower triangular part
    """
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i >= j:
                A[i, j] = abs(i - j + 1) ** (-p)
    # Make A as symmetric
    A += A.T - np.diag(A.diagonal())

    return A


def generate_FFT_system(n: int) -> np.ndarray:
    """
        A_n = [
        a_0, a_{-1}, ..., a_{2 - n}, a_{1 - n},
        a_1, a_0, a{-1}, ..., a_{2 - n},
        ...
        a_{n-1}, a_{n-2}, ..., a_1, a_0
    ]
    A:SPD
    a_k = 1 / (2pi) * integral_-pi^pi f(theta) e^(-i * k * theta) dtheta
    f(theta) = theta^4 + 1
    """
    f = lambda theta: theta ** 4.0 + 1
    thetas = np.linspace(-np.pi, np.pi, n, endpoint=False)
    first_col = (np.fft.fft(f(thetas))) / np.float64(n)
    A = scipy.linalg.toeplitz(first_col.real, first_col.real)

    return A