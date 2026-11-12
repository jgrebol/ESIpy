import pickle, sys, time
from tmat import build
S = eval(sys.argv[1]); t = time.time()
keys, trans, v0, w = build(S)
pickle.dump((len(keys), trans, [int(x) for x in v0], [int(x) for x in w]), open("T_%s.pkl" % "_".join(map(str, S)), "wb"))
print(len(keys), len(trans), time.time() - t)
