import numpy as np, sys, time
from tensor_rank import ptensor, als
rng = np.random.default_rng(int(sys.argv[2]))
T = ptensor(5); s = int(sys.argv[1])
t = time.time()
res = [als(T, s, 5000, rng) for _ in range(6)]
print('m=5 s=%d residuals over 6 ALS restarts x 5000 sweeps:' % s, ' '.join('%.2e' % x for x in sorted(res)), '%.1fs' % (time.time() - t))
