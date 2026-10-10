"""
实验一：线性方程组求解
求解 7×7 线性方程组 Ax = b，比较五种方法的求解结果
"""

import sys
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from solvers import gauss_elim, cholesky_solve, doolittle_solve, jacobi, gauss_seidel

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULT_DIR, exist_ok=True)


def main():
    # === 系数矩阵 A 和右端向量 b ===
    A = np.array([
        [10,  1,  2, -1,  1,  0,  1],
        [ 1, 11, -1,  2,  0,  1, -1],
        [ 2, -1, 12,  1,  2, -1,  0],
        [-1,  2,  1, 10, -1,  1,  1],
        [ 1,  0,  2, -1, 11,  2, -1],
        [ 0,  1, -1,  1,  2, 10,  1],
        [ 1, -1,  0,  1, -1,  1,  9]
    ], dtype=np.float64)

    b = np.array([14, 13, 15, 13, 14, 14, 10], dtype=np.float64)
    x_star = np.ones(7, dtype=np.float64)
    x0 = np.zeros(7, dtype=np.float64)

    print("=" * 60)
    print("实验一：线性方程组求解")
    print("=" * 60)
    print(f"\n系数矩阵 A:\n{A}")
    print(f"\n右端向量 b = {b}")
    print(f"精确解 x* = {x_star}")
    print(f"初始值 x0 = {x0}")

    # 矩阵性质
    print(f"\nA 对称: {np.allclose(A, A.T)}")
    eigvals = np.linalg.eigvalsh(A)
    print(f"A 特征值范围: [{eigvals.min():.4f}, {eigvals.max():.4f}]")
    print(f"A 正定: {eigvals.min() > 0}")
    diag_abs = np.abs(np.diag(A))
    print(f"A 严格对角占优: {np.all(diag_abs > np.sum(np.abs(A), axis=1) - diag_abs)}")

    # === 直接法 ===
    print("\n" + "=" * 60)
    print("直接法求解结果")
    print("=" * 60)

    x_gauss = gauss_elim(A, b)
    print(f"\n[1] 高斯消去法: x = {x_gauss}")
    print(f"    误差 = {np.linalg.norm(x_gauss - x_star):.2e}")

    x_chol = cholesky_solve(A, b)
    print(f"\n[2] Cholesky 分解法: x = {x_chol}")
    print(f"    误差 = {np.linalg.norm(x_chol - x_star):.2e}")

    x_dool = doolittle_solve(A, b)
    print(f"\n[3] Doolittle 分解法: x = {x_dool}")
    print(f"    误差 = {np.linalg.norm(x_dool - x_star):.2e}")

    # === 迭代法 ===
    print("\n" + "=" * 60)
    print("迭代法求解结果")
    print("=" * 60)

    tol = 1e-10
    max_iter = 10000

    x_jac, hist_jac, conv_jac, n_jac = jacobi(A, b, x0, tol, max_iter, x_star, record=True)
    print(f"\n[4] Jacobi 迭代: 收敛={conv_jac}, 迭代次数={n_jac}")
    print(f"    最终解 x = {x_jac}")
    print(f"    最终残差 = {hist_jac[-1][2]:.2e}, 最终误差 = {hist_jac[-1][3]:.2e}")

    x_gs, hist_gs, conv_gs, n_gs = gauss_seidel(A, b, x0, tol, max_iter, x_star, record=True)
    print(f"\n[5] Gauss-Seidel 迭代: 收敛={conv_gs}, 迭代次数={n_gs}")
    print(f"    最终解 x = {x_gs}")
    print(f"    最终残差 = {hist_gs[-1][2]:.2e}, 最终误差 = {hist_gs[-1][3]:.2e}")

    # === 保存直接法结果到 CSV ===
    with open(os.path.join(RESULT_DIR, 'direct_methods_results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'error'])
        for name, xs in [('Gauss', x_gauss), ('Cholesky', x_chol), ('Doolittle', x_dool)]:
            w.writerow([name] + list(xs) + [np.linalg.norm(xs - x_star)])

    # === 保存迭代历史到 CSV ===
    for name, hist in [('jacobi', hist_jac), ('gauss_seidel', hist_gs)]:
        with open(os.path.join(RESULT_DIR, f'{name}_history.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['k', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'residual', 'error'])
            for k, xk, res, err in hist:
                w.writerow([k] + list(xk) + [res, err])

    # === 可视化：误差曲线 ===
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.semilogy([h[0] for h in hist_jac], [h[3] for h in hist_jac], 'b-o', ms=3, label='Jacobi')
    ax.semilogy([h[0] for h in hist_gs], [h[3] for h in hist_gs], 'r-s', ms=3, label='Gauss-Seidel')
    ax.set_xlabel('Iteration k', fontsize=13)
    ax.set_ylabel(r'Error $e_k = \|x^{(k)} - x^*\|_2$', fontsize=13)
    ax.set_title('Jacobi vs Gauss-Seidel Iteration Error', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'iteration_error_comparison.png'), dpi=150)
    plt.close()

    print(f"\n结果已保存至: {RESULT_DIR}")
    print("=" * 60)
    print("实验一完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
