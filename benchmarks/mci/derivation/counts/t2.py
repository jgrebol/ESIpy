from lib import *
from tm import *
import time
for S in ([1], [1, 2], [1, 3], [1, 2, 3], [1, 2, 3, 4], [1,3,5]):
    s = max(S)
    for n in range(2 * s + 1, 15):
        a = ham_count(n, S); b = hk_directed(circ_adj(n, S))
        assert a == b, (S, n, a, b)
print("ham ok")
for F in ([1], [1, 2], [1,2,3]):
    for n in range(2 * max(F) + 1, 15):
        comp = [o for o in range(1, n // 2 + 1) if o not in F]
        a = ham_complement(n, F); b = hk_directed(circ_adj(n, comp))
        assert a == b, (F, n, a, b)
print("complement ok")
t=time.time(); print(ham_count(40,[1,2,3]), time.time()-t)
t=time.time(); print(ham_count(30,[1,2,3,4]), time.time()-t)
