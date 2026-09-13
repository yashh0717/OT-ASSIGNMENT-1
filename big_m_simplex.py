import numpy as np


def solve_furniture_lp(M=1e4):
    # Coefficients: 2 variables (T, C), 2 slack (<=), 3 surplus (>=), 3 artificial, 1 RHS
    # Order: [T, C, s1, s2, e1, e2, e3, a1, a2, a3, RHS]
    constraints = [
        [2, 1,  1, 0,  0,  0,  0, 0, 0, 0, 100],  # Cutting <= 100
        [1, 3,  0, 1,  0,  0,  0, 0, 0, 0, 120],  # Finishing <= 120
        [1, 1,  0, 0, -1,  0,  0, 1, 0, 0,  45],  # Material >= 45
        [1, 0,  0, 0,  0, -1,  0, 0, 1, 0,  15],  # Tables >= 15
        [0, 1,  0, 0,  0,  0, -1, 0, 0, 1,  10],  # Chairs >= 10
    ]
    tab = np.array(constraints, dtype=float)
    basis = [2, 3, 7, 8, 9]  # Initial basic variables: s1, s2, a1, a2, a3
    m = len(tab)

    # Objective row: Maximize 70T + 50C -> -70T - 50C + M*(a1 + a2 + a3)
    z_row = np.zeros(tab.shape[1])
    z_row[0] = -70.0
    z_row[1] = -50.0
    for r in range(2, 5):
        z_row -= M * tab[r]
    tab = np.vstack([tab, z_row])

    # Simplex loop
    for _ in range(50):
        rc = tab[-1, :-1]
        if np.min(rc) >= -1e-5:
            break

        p_col = int(np.argmin(rc))

        col = tab[:m, p_col]
        rhs = tab[:m, -1]
        ratios = [rhs[i] / col[i] if col[i] > 1e-6 else np.inf for i in range(m)]

        if min(ratios) == np.inf:
            return None

        p_row = int(np.argmin(ratios))

        tab[p_row] /= tab[p_row, p_col]
        for i in range(m + 1):
            if i != p_row:
                tab[i] -= tab[i, p_col] * tab[p_row]

        basis[p_row] = p_col

    # Solution read-out
    sol = np.zeros(tab.shape[1] - 1)
    for r, b in enumerate(basis):
        sol[b] = tab[r, -1]

    return {
        "tables": sol[0],
        "chairs": sol[1],
        "profit": tab[-1, -1]
    }


if __name__ == "__main__":
    res = solve_furniture_lp()

    if res:
        print(f"Make Tables  : {res['tables']:.1f}")
        print(f"Make Chairs  : {res['chairs']:.1f}")
        print(f"Total Profit : ${res['profit']:,.2f}")
    else:
        print("Infeasible or unbounded problem.")