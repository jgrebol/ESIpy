import json, sympy, mpmath
from t9 import powersums
x = sympy.symbols('x'); g = json.load(open("growth.json"))
for S in ("[1, 2]", "[1, 3]", "[1, 2, 3]", "[1, 2, 3, 4]"):
    C = json.load(open("rec_%s.json" % "_".join(S.strip('[]').split(', '))))['C']; L = len(C) - 1
    p = sympy.Poly(sum(c * x ** (L - i) for i, c in enumerate(C)), x)
    facs = sympy.factor_list(p)[1]
    best = None
    for fac, m in facs:
        r = max(abs(complex(z)) for z in sympy.Poly(fac, x).nroots(n=20, maxsteps=500))
        if best is None or r > best[0]: best = (r, fac, m)
    co = [int(c) for c in sympy.Poly(best[1], x).all_coeffs()]
    ps = powersums(co, 100); seq = {int(k): v for k, v in g[S]["seq"].items()}
    rem = {n: seq[n] - 2 * ps[n] for n in seq}
    print(S, "dominant factor:", best[1].as_expr(), "mult", best[2])
    print("   H_dir(n) - 2*P(n), n=%d..:" % min(seq), [rem[n] for n in sorted(rem)][:14])
    print("   relative remainder at n=40,60,80: %.2e %.2e %.2e" % tuple(rem[n] / seq[n] for n in (40, 60, 80)))
