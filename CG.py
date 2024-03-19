import numpy as np
import time


def conjugate_gradient(A: np.ndarray, b: np.ndarray, max_iterations=10000, tol=1e-6) -> np.ndarray:
    start_time = time.perf_counter()
    x = np.zeros_like(b, dtype=np.float64)
    r = b - A @ x
    p = r.copy()

    r0 = r.copy()
    r0_norm = np.linalg.norm(r0)

    for j in range(max_iterations):
        q = A @ p
        alpha = np.dot(r, r) / np.dot(p, q)
        x = x + alpha * p
        r_new = r - alpha * q
        if np.linalg.norm(r_new) / r0_norm < tol:
            print("Number of iterations:", j + 1)
            print("Elapsed time:", time.perf_counter() - start_time, "s")
            return x
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
    print("Warning: Solution did not converge after {} iterations".format(max_iterations))
    print("Best approximation obtained within tolerance.")
    print("Elapsed time:", time.perf_counter() - start_time, "s")
    return x