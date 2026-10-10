"""
实验二：不同规模下的求解效率对比
对 n = 1000, 2000, ..., 10000 维的随机正定矩阵，比较五种方法的求解时间
每种维度生成 5 组方程，取平均求解时间
"""

import sys
import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from solvers import gauss_elim, cholesky_solve, doolittle_solve, jacobi, gauss_seidel

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULT_DIR, exist_ok=True)


def generate_spd_matrix(n, seed):
    """
    随机生成对称正定矩阵 A 和向量 b
    生成对称矩阵后增强对角线使其严格对角占优，保证 SPD 且迭代收敛
    """
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    A = (B + B.T) / 2
    # 增加对角线使其严格对角占优
    off_diag_sum = np.sum(np.abs(A), axis=1) - np.abs(np.diag(A))
    np.fill_diagonal(A, off_diag_sum + rng.uniform(1.0, 2.0, n))
    b = rng.standard_normal(n)
    return A, b


def main():
    print("=" * 60)
    print("实验二：不同规模下的求解效率对比")
    print("=" * 60)

    dimensions = list(range(1000, 10001, 1000))
    n_groups = 5
    methods = ['Gauss', 'Cholesky', 'Doolittle', 'Jacobi', 'Gauss-Seidel']

    times = {m: {n: [] for n in dimensions} for m in methods}
    iters = {'Jacobi': {n: [] for n in dimensions},
             'Gauss-Seidel': {n: [] for n in dimensions}}
    converged = {'Jacobi': {n: [] for n in dimensions},
                 'Gauss-Seidel': {n: [] for n in dimensions}}

    tol = 1e-10
    max_iter = 10000

    for n in dimensions:
        print(f"\n--- n = {n} ---", flush=True)
        # 生成 5 组方程（不计入计时）
        problems = [generate_spd_matrix(n, n * 100 + s) for s in range(n_groups)]

        for s, (A, b) in enumerate(problems):
            t0 = time.perf_counter()
            _ = gauss_elim(A, b)
            t_gauss = time.perf_counter() - t0
            times['Gauss'][n].append(t_gauss)

            t0 = time.perf_counter()
            _ = cholesky_solve(A, b)
            t_chol = time.perf_counter() - t0
            times['Cholesky'][n].append(t_chol)

            t0 = time.perf_counter()
            _ = doolittle_solve(A, b)
            t_dool = time.perf_counter() - t0
            times['Doolittle'][n].append(t_dool)

            t0 = time.perf_counter()
            _, _, conv_j, nit_j = jacobi(A, b, None, tol, max_iter, record=False)
            t_jac = time.perf_counter() - t0
            times['Jacobi'][n].append(t_jac)
            iters['Jacobi'][n].append(nit_j)
            converged['Jacobi'][n].append(conv_j)

            t0 = time.perf_counter()
            _, _, conv_g, nit_g = gauss_seidel(A, b, None, tol, max_iter, record=False)
            t_gs = time.perf_counter() - t0
            times['Gauss-Seidel'][n].append(t_gs)
            iters['Gauss-Seidel'][n].append(nit_g)
            converged['Gauss-Seidel'][n].append(conv_g)

            print(f"  group {s+1}: "
                  f"Gauss={t_gauss:.3f}s  Chol={t_chol:.3f}s  Dool={t_dool:.3f}s  "
                  f"Jac={t_jac:.3f}s(it={nit_j},{'Y' if conv_j else 'N'})  "
                  f"GS={t_gs:.3f}s(it={nit_g},{'Y' if conv_g else 'N'})", flush=True)

    # === 计算平均时间 ===
    avg_times = {m: {n: np.mean(times[m][n]) for n in dimensions} for m in methods}

    # === 保存详细耗时到 CSV ===
    with open(os.path.join(RESULT_DIR, 'detailed_times.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n', 'group', 'Gauss', 'Cholesky', 'Doolittle',
                     'Jacobi', 'Gauss-Seidel', 'Jacobi_iter', 'GS_iter',
                     'Jacobi_converged', 'GS_converged'])
        for n in dimensions:
            for s in range(n_groups):
                w.writerow([n, s + 1,
                            times['Gauss'][n][s], times['Cholesky'][n][s],
                            times['Doolittle'][n][s], times['Jacobi'][n][s],
                            times['Gauss-Seidel'][n][s],
                            iters['Jacobi'][n][s], iters['Gauss-Seidel'][n][s],
                            converged['Jacobi'][n][s], converged['Gauss-Seidel'][n][s]])

    # === 保存平均耗时到 CSV ===
    with open(os.path.join(RESULT_DIR, 'average_times.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n'] + methods)
        for n in dimensions:
            w.writerow([n] + [avg_times[m][n] for m in methods])

    # === 可视化：五条时间曲线 ===
    fig, ax = plt.subplots(figsize=(12, 8))
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for i, m in enumerate(methods):
        ax.plot(dimensions, [avg_times[m][n] for n in dimensions],
                marker=markers[i], color=colors[i], label=m, linewidth=2, markersize=6)
    ax.set_xlabel('Matrix Dimension n', fontsize=13)
    ax.set_ylabel(r'Average Solving Time $\bar{T}_m(n)$ (s)', fontsize=13)
    ax.set_title('Solving Time Comparison of Five Methods', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(dimensions)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'time_comparison.png'), dpi=150)
    plt.close()

    # 对数纵轴版本
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, m in enumerate(methods):
        ax.semilogy(dimensions, [avg_times[m][n] for n in dimensions],
                     marker=markers[i], color=colors[i], label=m, linewidth=2, markersize=6)
    ax.set_xlabel('Matrix Dimension n', fontsize=13)
    ax.set_ylabel(r'Average Solving Time (s, log scale)', fontsize=13)
    ax.set_title('Solving Time Comparison (Log Scale)', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.set_xticks(dimensions)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'time_comparison_log.png'), dpi=150)
    plt.close()

    # === 打印收敛性信息 ===
    print("\n" + "=" * 60)
    print("收敛性信息")
    print("=" * 60)
    for m in ['Jacobi', 'Gauss-Seidel']:
        print(f"\n{m}:")
        for n in dimensions:
            convs = converged[m][n]
            print(f"  n={n:>6}: {sum(convs)}/{n_groups} converged, "
                  f"avg iterations = {np.mean(iters[m][n]):.1f}")

    # === 打印平均时间表 ===
    print("\n" + "=" * 60)
    print("平均求解时间 (秒)")
    print("=" * 60)
    header = f"{'n':>8}" + "".join(f"{m:>14}" for m in methods)
    print(header)
    print("-" * len(header))
    for n in dimensions:
        print(f"{n:>8}" + "".join(f"{avg_times[m][n]:>14.4f}" for m in methods))

    print(f"\n结果已保存至: {RESULT_DIR}")
    print("=" * 60)
    print("实验二完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
