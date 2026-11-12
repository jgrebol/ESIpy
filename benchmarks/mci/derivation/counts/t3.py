from tm import *; from bm import *; import time, sys
S = eval(sys.argv[1]); N = int(sys.argv[2]); step = int(sys.argv[3]) if len(sys.argv)>3 else 1
s = max(S); n0 = 2*s+1
t=time.time()
ns = list(range(n0 + (n0 % step if step==2 and False else 0), N+1))
if step == 2: ns = [n for n in ns if n % 2 == int(sys.argv[4])]
seq = [ham_count(n, S) for n in ns]
C, L = berlekamp_massey(seq)
print(S, "n:", ns[0], "..", ns[-1], "terms", len(seq), "order", L, "time %.1f" % (time.time()-t))
print("C =", C)
print("seq =", seq)
