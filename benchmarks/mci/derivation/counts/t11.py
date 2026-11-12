import json, sympy
from t9 import powersums
x = sympy.symbols('x'); g = json.load(open("growth.json"))
for S in ("[1, 3]", "[1, 2, 3]"):
    C = json.load(open("rec_%s.json" % "_".join(S.strip('[]').split(', '))))['C']; L = len(C) - 1
    p = sympy.Poly(sum(c * x ** (L - i) for i, c in enumerate(C)), x)
    seq = {int(k): v for k, v in g[S]["seq"].items()}
    basis, names = [], []
    for fac, m in sympy.factor_list(p)[1]:
        co = [int(c) for c in sympy.Poly(fac, x).all_coeffs()]; k = len(co) - 1
        ps = powersums(co, 120)
        for i in range(m):
            for j in range(k):
                basis.append(lambda n, ps=ps, i=i, j=j: n ** i * ps[n + j]); names.append("n^%d*P_{%s}(n+%d)" % (i, fac.as_expr(), j))
    ns = sorted(seq)
    M = sympy.Matrix([[b(n) for b in basis] for n in ns]); bb = sympy.Matrix([seq[n] for n in ns])
    sol, par = M.gauss_jordan_solve(bb)
    print(S, "dim", len(basis), "free params", par.shape)
    for nm, c in zip(names, sol):
        if c != 0: print("   %s  %s" % (c, nm))
