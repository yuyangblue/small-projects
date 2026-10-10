import numpy as np
from scipy.linalg import solve_triangular


def forward_sub(L, b):
    return solve_triangular(L, b, lower=True)


def back_sub(U, b):
    return solve_triangular(U, b, lower=False)


def gauss_elim(A, b):
    n = len(b)
    A = np.array(A, dtype=np.float64).copy()
    b = np.array(b, dtype=np.float64).copy()

    for k in range(n):
        # 列主元
        pivot = np.argmax(np.abs(A[k:, k])) + k
        if pivot != k:
            A[[k, pivot]] = A[[pivot, k]]
            b[[k, pivot]] = b[[pivot, k]]
        factors = A[k + 1:, k] / A[k, k]
        A[k + 1:, k:] -= np.outer(factors, A[k, k:])
        b[k + 1:] -= factors * b[k]

    x = back_sub(A, b)
    return x


def cholesky_solve(A, b):
    n = len(b)
    A = np.array(A, dtype=np.float64).copy()
    b = np.array(b, dtype=np.float64).copy()

    L = np.zeros((n, n))
    for j in range(n):
        s = A[j, j] - L[j, :j] @ L[j, :j]
        L[j, j] = np.sqrt(s)
        L[j + 1:, j] = (A[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]

    y = forward_sub(L, b)
    x = back_sub(L.T, y)
    return x


def doolittle_solve(A, b):
    n = len(b)
    A = np.array(A, dtype=np.float64).copy()
    b = np.array(b, dtype=np.float64).copy()

    L = np.eye(n)
    U = np.zeros((n, n))
    for k in range(n):
        U[k, k:] = A[k, k:] - L[k, :k] @ U[:k, k:]
        if k < n - 1:
            L[k + 1:, k] = (A[k + 1:, k] - L[k + 1:, :k] @ U[:k, k]) / U[k, k]

    z = forward_sub(L, b)
    x = back_sub(U, z)
    return x


def jacobi(A, b, x0=None, tol=1e-10, max_iter=10000, x_star=None):
    n = len(b)
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=np.float64).copy()

    D = np.diag(A).copy()
    R = A - np.diag(D)
    b_norm = np.linalg.norm(b)
    history = []
    converged = False
    n_iter = 0

    for k in range(max_iter):
        x_new = (b - R @ x) / D
        r = b - A @ x_new
        res_norm = np.linalg.norm(r)
        err = np.linalg.norm(x_new - x_star) if x_star is not None else float('nan')
        history.append((k, x_new.copy(), res_norm, err))
        x = x_new
        n_iter = k + 1
        if res_norm / b_norm <= tol:
            converged = True
            break

    return x, history, converged, n_iter


def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=10000, x_star=None):
    n = len(b)
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=np.float64).copy()

    L = np.tril(A)
    U = np.triu(A, 1)
    b_norm = np.linalg.norm(b)
    history = []
    converged = False
    n_iter = 0

    for k in range(max_iter):
        x_new = forward_sub(L, b - U @ x)
        r = b - A @ x_new
        res_norm = np.linalg.norm(r)
        err = np.linalg.norm(x_new - x_star) if x_star is not None else float('nan')
        history.append((k, x_new.copy(), res_norm, err))
        x = x_new
        n_iter = k + 1
        if res_norm / b_norm <= tol:
            converged = True
            break

    return x, history, converged, n_iter
