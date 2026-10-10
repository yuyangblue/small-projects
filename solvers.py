"""
线性方程组求解器模块
包含五种求解方法：高斯消去、Cholesky 分解、Doolittle 分解、Jacobi 迭代、Gauss-Seidel 迭代
"""

import numpy as np

# 尝试使用 scipy 的三角求解（基于 LAPACK，更快）；无 scipy 时用纯 numpy 实现
try:
    from scipy.linalg import solve_triangular as _solve_tri

    def _forward_sub(L, b, unit_diag=False):
        """前代法：解 Ly = b"""
        return _solve_tri(L, b, lower=True, unit_diagonal=unit_diag)

    def _back_sub(U, b):
        """回代法：解 Ux = b"""
        return _solve_tri(U, b, lower=False)
except ImportError:
    def _forward_sub(L, b, unit_diag=False):
        n = len(b)
        y = np.zeros(n)
        for i in range(n):
            y[i] = b[i] - L[i, :i] @ y[:i]
            if not unit_diag:
                y[i] /= L[i, i]
        return y

    def _back_sub(U, b):
        n = len(b)
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
        return x


def gauss_elim(A, b):
    """高斯消去法（带列主元选取）"""
    n = len(b)
    A = np.array(A, dtype=np.float64).copy()
    b = np.array(b, dtype=np.float64).copy()

    for k in range(n):
        # 列主元选取
        pivot = np.argmax(np.abs(A[k:, k])) + k
        if pivot != k:
            A[[k, pivot]] = A[[pivot, k]]
            b[[k, pivot]] = b[[pivot, k]]
        # 消去第 k 列
        factors = A[k + 1:, k] / A[k, k]
        A[k + 1:, k:] -= np.outer(factors, A[k, k:])
        b[k + 1:] -= factors * b[k]

    # 回代
    x = _back_sub(A, b)
    return x


def cholesky_solve(A, b):
    """Cholesky 分解法：A = LL^T"""
    n = len(b)
    A = np.array(A, dtype=np.float64).copy()
    b = np.array(b, dtype=np.float64).copy()

    L = np.zeros((n, n))
    for j in range(n):
        s = A[j, j] - L[j, :j] @ L[j, :j]
        if s <= 0:
            raise ValueError(f"矩阵非正定，Cholesky 分解失败")
        L[j, j] = np.sqrt(s)
        L[j + 1:, j] = (A[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]

    y = _forward_sub(L, b)
    x = _back_sub(L.T, y)
    return x


def doolittle_solve(A, b):
    """Doolittle 分解法（LU 分解）：A = LU"""
    n = len(b)
    A = np.array(A, dtype=np.float64).copy()
    b = np.array(b, dtype=np.float64).copy()

    L = np.eye(n)
    U = np.zeros((n, n))
    for k in range(n):
        U[k, k:] = A[k, k:] - L[k, :k] @ U[:k, k:]
        if k < n - 1:
            L[k + 1:, k] = (A[k + 1:, k] - L[k + 1:, :k] @ U[:k, k]) / U[k, k]

    z = _forward_sub(L, b, unit_diag=True)
    x = _back_sub(U, z)
    return x


def jacobi(A, b, x0=None, tol=1e-10, max_iter=10000, x_star=None, record=True):
    """Jacobi 迭代法：x^{(k+1)} = D^{-1}(b - (L+U)x^{(k)})"""
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
        if not np.all(np.isfinite(x_new)):
            break
        r = b - A @ x_new
        res_norm = np.linalg.norm(r)
        rel_res = res_norm / b_norm if b_norm > 0 else res_norm
        if record:
            err = np.linalg.norm(x_new - x_star) if x_star is not None else float('nan')
            history.append((k, x_new.copy(), res_norm, err))
        x = x_new
        n_iter = k + 1
        if rel_res <= tol:
            converged = True
            break

    return x, history, converged, n_iter


def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=10000, x_star=None, record=True):
    """Gauss-Seidel 迭代法：(D+L)x^{(k+1)} = b - Ux^{(k)}"""
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
        rhs = b - U @ x
        x_new = _forward_sub(L, rhs)
        if not np.all(np.isfinite(x_new)):
            break
        r = b - A @ x_new
        res_norm = np.linalg.norm(r)
        rel_res = res_norm / b_norm if b_norm > 0 else res_norm
        if record:
            err = np.linalg.norm(x_new - x_star) if x_star is not None else float('nan')
            history.append((k, x_new.copy(), res_norm, err))
        x = x_new
        n_iter = k + 1
        if rel_res <= tol:
            converged = True
            break

    return x, history, converged, n_iter
