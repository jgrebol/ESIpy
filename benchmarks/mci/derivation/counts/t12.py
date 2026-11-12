"""lambda and C for S=[1,3,5] by float power iteration on the transfer matrix (H(n)=w T^(n-15) v0)."""
import pickle, numpy as np, math
D, trans, v0, w = pickle.load(open("T_1_3_5.pkl", "rb"))
fr = np.array([t[0] for t in trans]); to = np.array([t[1] for t in trans]); cc = np.array([t[2] for t in trans], dtype=float)
vec = np.array(v0, dtype=float); ww = np.array(w, dtype=float); logs = 0.0; s = 5
vals = {}
for k in range(0, 1201):
    n = 3 * s + k
    h = float(ww @ vec)
    vals[n] = (math.log(h) + logs) if h > 0 else None
    nv = np.zeros(D); np.add.at(nv, to, cc * vec[fr]); nrm = nv.sum(); vec = nv / nrm; logs += math.log(nrm)
for n in (16, 18, 20, 22, 30, 31, 40):
    print(n, round(math.exp(vals[n])) if n < 60 else vals[n])
for n in (600, 800, 1000, 1200):
    lam = math.exp((vals[n] - vals[n - 2]) / 2); print(n, "lambda(even) %.12f" % lam, "lambda(odd) %.12f" % math.exp((vals[n - 1] - vals[n - 3]) / 2))
lam = math.exp((vals[1200] - vals[1198]) / 2)
print("C even:", math.exp(vals[1200] - 1200 * math.log(lam)), "C odd:", math.exp(vals[1199] - 1199 * math.log(lam)))
