import numpy as np
from pyscf import gto

def get_sdiag(l):
    mol = gto.M(atom='H 0 0 0', basis={'H': [[l, [1.0, 1.0]]]}, cart=True, spin=1).build()
    s = mol.intor('int1e_ovlp')
    return np.sqrt(np.diag(s))

for l in range(2, 6):
    factors = get_sdiag(l)
    print(f"L={l}: {np.round(factors**2, 10)}")
