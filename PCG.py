import numpy as np
import time
import math


def preconditioned_conjugate_gradient(A: np.ndarray, b: np.ndarray, preconditioner_fun, max_iterations=10000, tol=1e-6) -> np.ndarray:
    """
    Solve Ax = b using the preconditioned conjugate gradient method.
    Use CG in M-inner product
    :param M: preconditioner
    """
    start_time = time.time()

    C_first_column = preconditioner_fun(A)
    D_inv = 1 / np.fft.fft(C_first_column)

    x = np.zeros_like(b, dtype=np.float64)
    r = b - A @ x
    # Solve Mz = r
    z = np.fft.ifft(D_inv * np.fft.fft(r)).real
    p = z.copy()

    r0 = r.copy()
    r0_norm = np.linalg.norm(r0)

    for j in range(max_iterations):
        q = A @ p
        alpha = np.dot(r, z) / np.dot(p, q)
        x = x + alpha * p
        r_new = r - alpha * q
        if np.linalg.norm(r_new) / r0_norm < tol:
            print("Number of iterations:", j + 1)
            print("Elapsed time:", time.time() - start_time, "s")
            return x
        # Solve Mz = r
        z_new = np.fft.ifft(D_inv * np.fft.fft(r_new)).real
        beta = np.dot(r_new, z_new) / np.dot(r, z)
        p = z_new + beta * p
        r = r_new
        z = z_new
    print("Warning: Solution did not converge after {} iterations".format(max_iterations))
    print("Best approximation obtained within tolerance.")
    print("Elapsed time:", time.time() - start_time, "s")
    return x


def get_A_index(k, n) -> tuple:
    """
    This function is used to get the location of the kth element in the A matrix.
    """
    # First row
    if k <= 0:
        row = 0
        col = -k
    # First column
    elif k > 0:
        row = k
        col = 0
    # This is in T_chan's case if k - n = n or -n in this case we use k = 0
    if k == n or k == -n:
        row = col = 0
    return row, col


def G_Strang_first_column(A: np.ndarray) -> np.ndarray :
    n = A.shape[0]
    D = np.zeros(n, dtype=np.float64)
    for k in range(n):
        if k <= math.floor(n / 2):
            D[k] = A[get_A_index(k, n)]
        else:
            D[k] = A[get_A_index(k - n, n)]
    return D


def T_Chan_first_column(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    D = np.zeros(n, dtype=np.float64)
    for k in range(n):
        aj = A[get_A_index(k, n)]
        aj_minus_n = A[get_A_index(k - n, n)]
        D[k] = ((n - k) * aj + k * aj_minus_n) / n
    return D