"""Try closed forms H_dir(n) = sum of power sums of the factors of the char poly (exact rational fit, then checked)."""
import json, sympy
from fractions import Fraction
x = sympy.symbols('x')
def powersums(coeffs, N):
    # monic poly x^k + c1 x^{k-1}+...+ck ; Newton identities
    c = coeffs[1:]; k = len(c); p = [k]
    for m in range(1, N + 1):
        v = -m * c[m - 1] if m <= k else 0
        v -= sum(c[i - 1] * p[m - i] for i in range(1, min(m - 1, k) + 1))
        p.append(v)
    return p
def solve(rows, rhs):
    M = sympy.Matrix(rows); b = sympy.Matrix(rhs)
    sol = M.solve_least_squares(b) if M.rows > M.cols else M.solve(b)
    return sol
g = json.load(open("growth.json"))
for S, extra in (("[1, 2]", []), ("[1, 3]", []), ("[1, 2, 3]", [])):
    C = json.load(open("rec_%s.json" % "_".join(S.strip('[]').split(', '))))['C']; L = len(C) - 1
    p = sympy.Poly(sum(c * x ** (L - i) for i, c in enumerate(C)), x)
    seq = {int(k): v for k, v in g[S]["seq"].items()}
    basis = []; names = []
    for fac, m in sympy.factor_list(p)[1]:
        co = [int(c) for c in sympy.Poly(fac, x).all_coeffs()]
        ps = powersums(co, 100)
        for j in range(m):
            basis.append(lambda n, ps=ps, j=j: n ** j * ps[n]); names.append("n^%d*P[%s]" % (j, fac.as_expr()))
    ns = sorted(seq)
    rows = [[b(n) for b in basis] for n in ns]
    M = sympy.Matrix(rows); bb = sympy.Matrix([seq[n] for n in ns])
    try:
        sol, params = M.gauss_jordan_solve(bb)
        print(S, "EXACT FIT over n=%d..%d:" % (ns[0], ns[-1]))
        for nm, c in zip(names, sol): print("    %s * %s" % (c, nm))
    except ValueError:
        print(S, "no fit with power-sum basis (conjugate-dependent coefficients)")
