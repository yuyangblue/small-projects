import numpy as np


def forward_sub(L, b):
    L = np.asarray(L, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square matrix")

    n = L.shape[0]
    if b.ndim == 1:
        if b.shape[0] != n:
            raise ValueError("Dimension mismatch between L and b")
        x = np.empty(n, dtype=np.float64)
        for i in range(n):
            s = b[i] - L[i, :i] @ x[:i]
            if abs(L[i, i]) < np.finfo(np.float64).eps:
                raise np.linalg.LinAlgError("Matrix is singular or not invertible")
            x[i] = s / L[i, i]
        return x

    if b.ndim == 2:
        if b.shape[0] != n:
            raise ValueError("Dimension mismatch between L and b")
        x = np.empty_like(b, dtype=np.float64)
        for j in range(b.shape[1]):
            for i in range(n):
                s = b[i, j] - L[i, :i] @ x[:i, j]
                if abs(L[i, i]) < np.finfo(np.float64).eps:
                    raise np.linalg.LinAlgError("Matrix is singular or not invertible")
                x[i, j] = s / L[i, i]
        return x

    raise ValueError("b must be a 1D or 2D array")


def back_sub(U, b):
    U = np.asarray(U, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("U must be a square matrix")

    n = U.shape[0]
    if b.ndim == 1:
        if b.shape[0] != n:
            raise ValueError("Dimension mismatch between U and b")
        x = np.empty(n, dtype=np.float64)
        for i in range(n - 1, -1, -1):
            s = b[i] - U[i, i + 1:] @ x[i + 1:]
            if abs(U[i, i]) < np.finfo(np.float64).eps:
                raise np.linalg.LinAlgError("Matrix is singular or not invertible")
            x[i] = s / U[i, i]
        return x

    if b.ndim == 2:
        if b.shape[0] != n:
            raise ValueError("Dimension mismatch between U and b")
        x = np.empty_like(b, dtype=np.float64)
        for j in range(b.shape[1]):
            for i in range(n - 1, -1, -1):
                s = b[i, j] - U[i, i + 1:] @ x[i + 1:, j]
                if abs(U[i, i]) < np.finfo(np.float64).eps:
                    raise np.linalg.LinAlgError("Matrix is singular or not invertible")
                x[i, j] = s / U[i, i]
        return x

    raise ValueError("b must be a 1D or 2D array")


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


def is_positive_definite(A):
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if not np.allclose(A, A.T):
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def cholesky_solve(A, b):
    n = len(b)
    A = np.array(A, dtype=np.float64).copy()
    b = np.array(b, dtype=np.float64).copy()

    if not is_positive_definite(A):
        raise np.linalg.LinAlgError("A is not symmetric positive definite")

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
