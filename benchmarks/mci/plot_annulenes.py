"""
Plots of the annulene scan (scan_alg5.jsonl, scan_exact.jsonl): ring-normalised MCI and cost against ring size.
  python benchmarks/mci/plot_annulenes.py      -> plots/annulenes_mci.{png,pdf}
Ring-normalised MCI = sign(MCI) |MCI|^(1/n). Rerun whenever new rings are added.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
rows = {}
for name in ("scan_alg5.jsonl", "scan_exact.jsonl"):
    p = os.path.join(here, name)
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            rows.setdefault(r["n"], {}).update({k: v for k, v in r.items() if v is not None})
ns = sorted(rows)
norm = lambda v, n: np.sign(v) * abs(v) ** (1 / n)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for kind, marker, color in (("exact", "o", "#1f4e9e"), ("alg5", "s", "#d1495b")):
    for cls, fill in ((2, True), (0, False)):  # 4n+2 filled, 4n open
        pts = [(n, norm(rows[n][kind], n)) for n in ns if kind in rows[n] and n % 4 == cls]
        if pts:
            x, y = zip(*pts)
            ax1.plot(x, y, marker, ms=7 if kind == "exact" else 5, color=color, mfc=color if fill else "white",
                     label=f"{'exact MCI' if kind == 'exact' else 'mcialg=5, d=kmax'} ({'4n+2' if cls == 2 else '4n'})")
for kind, color in (("exact", "#1f4e9e"), ("alg5", "#d1495b")):
    x = [n for n in ns if kind in rows[n]]
    ax1.plot(x, [norm(rows[n][kind], n) for n in x], "-", color=color, lw=0.8, alpha=0.5)
ax1.axhline(0, color="gray", lw=0.6)
ax1.set_xlabel("ring size n (CnHn)")
ax1.set_ylabel(r"ring-normalised MCI  sign(MCI)$\,|$MCI$|^{1/n}$")
ax1.set_title("Annulenes, RHF/STO-3G, meta-Löwdin")
ax1.legend(fontsize=8, frameon=False)

for kind, key, color, label in (("exact", "exact_s", "#1f4e9e", "exact MCI"), ("alg5", "alg5_s", "#d1495b", "mcialg=5, d=kmax")):
    x = [n for n in ns if key in rows[n]]
    ax2.semilogy(x, [rows[n][key] for n in x], "o-", color=color, label=label, ms=5)
ax2.set_xlabel("ring size n")
ax2.set_ylabel("time (s, 6-8 cores)")
ax2.set_title("Cost with Glynn (memory constant)")
ax2.grid(True, which="both", lw=0.3, alpha=0.5)
ax2.legend(fontsize=8, frameon=False)
fig.tight_layout()
os.makedirs(os.path.join(here, "plots"), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(here, "plots", f"annulenes_mci.{ext}"), dpi=200)
print(f"{'n':>3s} {'exact':>14s} {'alg5':>14s} {'norm exact':>10s} {'norm alg5':>10s} {'alg5/exact':>11s} {'t exact':>9s} {'t alg5':>9s}")
for n in ns:
    r = rows[n]
    e, a = r.get("exact"), r.get("alg5")
    print(f"{n:3d} {e if e is not None else float('nan'):+14.6e} {a if a is not None else float('nan'):+14.6e} "
          f"{norm(e, n) if e is not None else float('nan'):10.5f} {norm(a, n) if a is not None else float('nan'):10.5f} "
          f"{100 * a / e if e and a else float('nan'):10.5f}% {r.get('exact_s', float('nan')):9.1f} {r.get('alg5_s', float('nan')):9.1f}")
