import numpy as np


def vam(costs, supply, demand):
    supply = list(supply)
    demand = list(demand)
    m, n = len(supply), len(demand)

    alloc = np.zeros((m, n))
    rows = set(range(m))
    cols = set(range(n))
    basis = set()

    while rows and cols:
        penalties = []

        for r in rows:
            vals = sorted(costs[r, c] for c in cols)
            diff = vals[1] - vals[0] if len(vals) > 1 else vals[0]
            penalties.append((diff, "r", r))

        for c in cols:
            vals = sorted(costs[r, c] for r in rows)
            diff = vals[1] - vals[0] if len(vals) > 1 else vals[0]
            penalties.append((diff, "c", c))

        _, axis, idx = max(penalties, key=lambda x: x[0])

        if axis == "r":
            r = idx
            c = min(cols, key=lambda j: costs[r, j])
        else:
            c = idx
            r = min(rows, key=lambda i: costs[i, c])

        qty = min(supply[r], demand[c])
        alloc[r, c] = qty
        basis.add((r, c))

        supply[r] -= qty
        demand[c] -= qty

        if demand[c] == 0 and len(cols) > 1:
            cols.remove(c)
        elif supply[r] == 0:
            rows.remove(r)
        else:
            cols.remove(c)

    # Pad basic cells if initial solution is degenerate
    for r in range(m):
        for c in range(n):
            if len(basis) == m + n - 1:
                break
            basis.add((r, c))

    return alloc, list(basis)


def find_loop(basis, start):
    nodes = set(basis) | {start}

    # Prune leaves that don't have pairs in both row and col
    while True:
        pruned = False
        for r, c in list(nodes):
            has_row = sum(1 for x, y in nodes if x == r) >= 2
            has_col = sum(1 for x, y in nodes if y == c) >= 2
            if not (has_row and has_col):
                nodes.remove((r, c))
                pruned = True
        if not pruned:
            break

    # Build alternating horizontal/vertical cycle
    loop = [start]
    move_horiz = True

    while True:
        curr = loop[-1]
        step = [
            pt for pt in nodes
            if pt != curr and ((pt[0] == curr[0]) if move_horiz else (pt[1] == curr[1]))
        ]
        if not step or step[0] == start:
            break
        loop.append(step[0])
        move_horiz = not move_horiz

    return loop


def modi(costs, alloc, basis):
    m, n = costs.shape

    while True:
        u = [None] * m
        v = [None] * n
        u[0] = 0

        # Calculate u and v values
        for _ in range(m + n):
            for r, c in basis:
                if u[r] is not None and v[c] is None:
                    v[c] = costs[r, c] - u[r]
                elif v[c] is not None and u[r] is None:
                    u[r] = costs[r, c] - v[c]

        # Find entering cell with most negative reduced cost
        best_delta, entering = 0, None
        for r in range(m):
            for c in range(n):
                if (r, c) not in basis:
                    delta = costs[r, c] - (u[r] + v[c])
                    if delta < best_delta:
                        best_delta = delta
                        entering = (r, c)

        if entering is None:
            break

        loop = find_loop(basis, entering)
        donor_cells = loop[1::2]
        theta = min(alloc[r, c] for r, c in donor_cells)

        for i, (r, c) in enumerate(loop):
            alloc[r, c] += theta if i % 2 == 0 else -theta

        leaving = next(pt for pt in donor_cells if alloc[pt[0], pt[1]] == 0)
        basis.remove(leaving)
        basis.append(entering)

    return alloc


if __name__ == "__main__":
    costs = np.array([
        [10,  2, 20, 11],
        [12,  7,  9, 20],
        [ 4, 14, 16, 18]
    ], dtype=float)

    supply = [15, 25, 10]
    demand = [5, 15, 15, 15]

    alloc, basis = vam(costs, supply, demand)
    print("VAM starting cost:", int(np.sum(alloc * costs)))

    alloc = modi(costs, alloc, basis)
    print("MODI optimal cost:", int(np.sum(alloc * costs)))
    print("Final allocation:\n", alloc)