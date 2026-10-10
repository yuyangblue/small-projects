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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def main():
    A = np.array([
        [10, 1, 2, -1, 1, 0, 1],
        [1, 11, -1, 2, 0, 1, -1],
        [2, -1, 12, 1, 2, -1, 0],
        [-1, 2, 1, 10, -1, 1, 1],
        [1, 0, 2, -1, 11, 2, -1],
        [0, 1, -1, 1, 2, 10, 1],
        [1, -1, 0, 1, -1, 1, 9]
    ], dtype=np.float64)

    b = np.array([14, 13, 15, 13, 14, 14, 10], dtype=np.float64)
    x_star = np.ones(7, dtype=np.float64)
    x0 = np.zeros(7, dtype=np.float64)

    print("实验一：线性方程组求解")
    print("A 对称:", np.allclose(A, A.T))
    print("A 正定:", np.all(np.linalg.eigvalsh(A) > 0))

    # 直接法
    x_gauss = gauss_elim(A, b)
    x_chol = cholesky_solve(A, b)
    x_dool = doolittle_solve(A, b)

    print("\n高斯消去法: x =", x_gauss, " 误差 =", np.linalg.norm(x_gauss - x_star))
    print("Cholesky分解: x =", x_chol, " 误差 =", np.linalg.norm(x_chol - x_star))
    print("Doolittle分解: x =", x_dool, " 误差 =", np.linalg.norm(x_dool - x_star))

    # 迭代法
    tol = 1e-10
    max_iter = 10000

    x_jac, hist_jac, conv_jac, n_jac = jacobi(A, b, x0, tol, max_iter, x_star)
    x_gs, hist_gs, conv_gs, n_gs = gauss_seidel(A, b, x0, tol, max_iter, x_star)

    print("\nJacobi迭代: 收敛=%s, 次数=%d" % (conv_jac, n_jac))
    print("  最终解 x =", x_jac)
    print("  残差 =", hist_jac[-1][2], " 误差 =", hist_jac[-1][3])

    print("\nGauss-Seidel迭代: 收敛=%s, 次数=%d" % (conv_gs, n_gs))
    print("  最终解 x =", x_gs)
    print("  残差 =", hist_gs[-1][2], " 误差 =", hist_gs[-1][3])

    # 保存直接法结果
    with open(os.path.join(RESULT_DIR, 'direct_methods_results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'error'])
        for name, xs in [('Gauss', x_gauss), ('Cholesky', x_chol), ('Doolittle', x_dool)]:
            w.writerow([name] + list(xs) + [np.linalg.norm(xs - x_star)])

    # 保存迭代历史
    for name, hist in [('jacobi', hist_jac), ('gauss_seidel', hist_gs)]:
        with open(os.path.join(RESULT_DIR, name + '_history.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['k', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'residual', 'error'])
            for k, xk, res, err in hist:
                w.writerow([k] + list(xk) + [res, err])

    # 画误差曲线
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.semilogy([h[0] for h in hist_jac], [h[3] for h in hist_jac], 'b-o', ms=3, label='Jacobi')
    ax.semilogy([h[0] for h in hist_gs], [h[3] for h in hist_gs], 'r-s', ms=3, label='Gauss-Seidel')
    ax.set_xlabel('迭代次数 k')
    ax.set_ylabel(r'误差 $e_k = \|x^{(k)} - x^*\|_2$')
    ax.set_title('Jacobi 与 Gauss-Seidel 迭代误差比较')
    ax.legend()
    ax.grid(True, linestyle='--')
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'iteration_error_comparison.png'), dpi=150)
    plt.close()

    print("\n结果保存至:", RESULT_DIR)


if __name__ == '__main__':
    main()
