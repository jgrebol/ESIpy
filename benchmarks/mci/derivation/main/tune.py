import time, sys, numpy as np
sys.path.insert(0, sys.argv[1])
from krylov_proto import krylov_exact, rand
for n, q, r in ((16, 56, 5), (16, 56, 14)):
    mats = rand(n, q, r, True)
    for nlow in (3, 4, 5, 6, 8, 10):
        t = time.time(); b = krylov_exact(mats, True, nlow); print(n, q, r, "nlow", nlow, f"{time.time()-t:.3f}s", flush=True)
