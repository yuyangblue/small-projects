import numpy as np


def validate_solution(A, b, x, tol=1e-8):
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("A must be a square matrix")
    n = A.shape[0]
    if b.shape != (n,):
        raise ValueError("b must be a vector matching A")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and non-negative")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("A and b must contain only finite values")
    if x.shape != (n,) or not np.all(np.isfinite(x)):
        return False, float('inf')

    residual_norm = np.linalg.norm(A @ x - b, ord=np.inf)
    scale = (np.linalg.norm(A, ord=np.inf) * np.linalg.norm(x, ord=np.inf)
             + np.linalg.norm(b, ord=np.inf))
    backward_error = residual_norm / scale if scale else residual_norm
    valid = np.isfinite(backward_error) and backward_error <= tol
    return bool(valid), float(backward_error)


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
        if A[k, k] == 0:
            raise np.linalg.LinAlgError("Matrix is singular")
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
    A = np.array(A, dtype=np.float64).copy()
    b = np.array(b, dtype=np.float64).copy()

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if not np.allclose(A, A.T):
        raise np.linalg.LinAlgError("A is not symmetric positive definite")

    n = A.shape[0]
    L = np.zeros((n, n))
    for j in range(n):
        s = A[j, j] - L[j, :j] @ L[j, :j]
        if not np.isfinite(s) or s <= 0:
            raise np.linalg.LinAlgError("A is not symmetric positive definite")
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
            if U[k, k] == 0:
                raise np.linalg.LinAlgError(
                    "Zero pivot in Doolittle decomposition; pivoting is required"
                )
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
    if np.any(D == 0):
        raise np.linalg.LinAlgError("Jacobi iteration requires non-zero diagonal entries")
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
        if res_norm <= tol * b_norm:
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
        if res_norm <= tol * b_norm:
            converged = True
            break

    return x, history, converged, n_iter
