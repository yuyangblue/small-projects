import sys
import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from solvers import (
    gauss_elim, cholesky_solve, doolittle_solve, jacobi, gauss_seidel,
    validate_solution,
)

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC', 'SimHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS'
]
plt.rcParams['axes.unicode_minus'] = False


def generate_spd_matrix(n, seed):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    A = (B + B.T) / 2
    # 增强对角线保证严格对角占优
    off_diag = np.sum(np.abs(A), axis=1) - np.abs(np.diag(A))
    np.fill_diagonal(A, off_diag + rng.uniform(1.0, 2.0, n))
    b = rng.standard_normal(n)
    return A, b


def main():
    print("实验二：不同规模下的求解效率对比")

    dimensions = list(range(1000, 10001, 1000))
    n_groups = 5
    methods = ['Gauss', 'Cholesky', 'Doolittle', 'Jacobi', 'Gauss-Seidel']

    times = {m: {n: [] for n in dimensions} for m in methods}
    iters = {'Jacobi': {n: [] for n in dimensions},
             'Gauss-Seidel': {n: [] for n in dimensions}}
    converged = {'Jacobi': {n: [] for n in dimensions},
                 'Gauss-Seidel': {n: [] for n in dimensions}}
    backward_errors = {m: {n: [] for n in dimensions} for m in methods}
    solution_valid = {m: {n: [] for n in dimensions} for m in methods}

    tol = 1e-10
    validation_tol = 1e-8
    max_iter = 10000

    for n in dimensions:
        print("\n n = %d" % n)

        for s in range(n_groups):
            A, b = generate_spd_matrix(n, n * 100 + s)

            t0 = time.perf_counter()
            x_gauss = gauss_elim(A, b)
            t_gauss = time.perf_counter() - t0
            times['Gauss'][n].append(t_gauss)

            t0 = time.perf_counter()
            x_chol = cholesky_solve(A, b)
            t_chol = time.perf_counter() - t0
            times['Cholesky'][n].append(t_chol)

            t0 = time.perf_counter()
            x_dool = doolittle_solve(A, b)
            t_dool = time.perf_counter() - t0
            times['Doolittle'][n].append(t_dool)

            t0 = time.perf_counter()
            x_jac, _, conv_j, nit_j = jacobi(A, b, None, tol, max_iter)
            t_jac = time.perf_counter() - t0
            times['Jacobi'][n].append(t_jac)
            iters['Jacobi'][n].append(nit_j)
            converged['Jacobi'][n].append(conv_j)

            t0 = time.perf_counter()
            x_gs, _, conv_g, nit_g = gauss_seidel(A, b, None, tol, max_iter)
            t_gs = time.perf_counter() - t0
            times['Gauss-Seidel'][n].append(t_gs)
            iters['Gauss-Seidel'][n].append(nit_g)
            converged['Gauss-Seidel'][n].append(conv_g)

            for method, x in [('Gauss', x_gauss), ('Cholesky', x_chol),
                              ('Doolittle', x_dool), ('Jacobi', x_jac),
                              ('Gauss-Seidel', x_gs)]:
                valid, error = validate_solution(A, b, x, validation_tol)
                solution_valid[method][n].append(valid)
                backward_errors[method][n].append(error)

            print("  group %d: Gauss=%.3f  Chol=%.3f  Dool=%.3f  "
                  "Jac=%.3f(it=%d)  GS=%.3f(it=%d)  valid=%d/%d" % (
                      s + 1, t_gauss, t_chol, t_dool, t_jac, nit_j, t_gs,
                      nit_g, sum(solution_valid[m][n][-1] for m in methods),
                      len(methods)))

    # 平均时间
    avg_times = {m: {n: np.mean(times[m][n]) for n in dimensions} for m in methods}

    # 保存详细耗时
    with open(os.path.join(RESULT_DIR, 'detailed_times.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n', 'group', 'Gauss', 'Cholesky', 'Doolittle',
                     'Jacobi', 'Gauss-Seidel', 'Jacobi_iter', 'GS_iter',
                     'Jacobi_converged', 'GS_converged']
                    + [m + '_backward_error' for m in methods]
                    + [m + '_valid' for m in methods])
        for n in dimensions:
            for s in range(n_groups):
                w.writerow([n, s + 1,
                            times['Gauss'][n][s], times['Cholesky'][n][s],
                            times['Doolittle'][n][s], times['Jacobi'][n][s],
                            times['Gauss-Seidel'][n][s],
                            iters['Jacobi'][n][s], iters['Gauss-Seidel'][n][s],
                            converged['Jacobi'][n][s], converged['Gauss-Seidel'][n][s]]
                           + [backward_errors[m][n][s] for m in methods]
                           + [solution_valid[m][n][s] for m in methods])

    # 保存平均耗时
    with open(os.path.join(RESULT_DIR, 'average_times.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n'] + methods)
        for n in dimensions:
            w.writerow([n] + [avg_times[m][n] for m in methods])

    # 画时间曲线
    fig, ax = plt.subplots(figsize=(12, 8))
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for i, m in enumerate(methods):
        ax.plot(dimensions, [avg_times[m][n] for n in dimensions],
                marker=markers[i], color=colors[i], label=m, linewidth=2, markersize=6)
    ax.set_xlabel('矩阵维度 n')
    ax.set_ylabel(r'平均求解时间 $\bar{T}_m(n)$ (秒)')
    ax.set_title('五种方法求解时间对比')
    ax.legend()
    ax.grid(True, linestyle='--')
    ax.set_xticks(dimensions)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'time_comparison.png'), dpi=150)
    plt.close()

    # 对数纵轴
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, m in enumerate(methods):
        ax.semilogy(dimensions, [avg_times[m][n] for n in dimensions],
                     marker=markers[i], color=colors[i], label=m, linewidth=2, markersize=6)
    ax.set_xlabel('矩阵维度 n')
    ax.set_ylabel('平均求解时间 (秒, 对数)')
    ax.set_title('五种方法求解时间对比 (对数纵轴)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--')
    ax.set_xticks(dimensions)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'time_comparison_log.png'), dpi=150)
    plt.close()

    # 打印收敛信息
    print("\n收敛信息:")
    for m in ['Jacobi', 'Gauss-Seidel']:
        for n in dimensions:
            convs = converged[m][n]
            print("  %s n=%d: %d/%d 收敛, 平均迭代 %d 次" % (
                m, n, sum(convs), n_groups, np.mean(iters[m][n])))

    print("\n解验证结果 (后向误差阈值 %.1e):" % validation_tol)
    for m in methods:
        for n in dimensions:
            valid_count = sum(solution_valid[m][n])
            max_error = max(backward_errors[m][n])
            print("  %s n=%d: %d/%d 通过, 最大后向误差 %.3e" % (
                m, n, valid_count, n_groups, max_error))

    # 打印平均时间表
    print("\n平均求解时间 (秒):")
    for n in dimensions:
        row = "n=%5d" % n
        for m in methods:
            row += "  %s=%.4f" % (m[:6], avg_times[m][n])
        print(row)

    print("\n结果保存至:", RESULT_DIR)


if __name__ == '__main__':
    main()
