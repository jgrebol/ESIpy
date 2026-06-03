import numpy as np
from pyscf import gto, scf
from esipy.readfchk import readfchk
import os

path = "../tests/FCHK/QCHEM/1_benzene_spherical.fchk"
mol_f, mf_f = readfchk(path)
S_f = mf_f.get_ovlp()

basis = 'cc-pVTZ'
atom = "C 0 0 0" # Single atom test
mol_p = gto.M(atom=atom, basis=basis).build()
S_p = mol_p.intor('int1e_ovlp')

# Carbon in cc-pVTZ: s(4), p(3), d(2), f(1) ?
# Actually cc-pVTZ for C is: [4s, 3p, 2d, 1f]
# PySCF: 14 AOs
s1_p = S_p[:14, :14]
s1_f = S_f[:14, :14]

print("PySCF sub-shell starts:")
# 1s(1), 2s(1), 3s(1), 4s(1) -> indices 0,1,2,3
# 2p(3), 3p(3), 4p(3) -> indices 4,5,6, 7,8,9, 10,11,12
# 3d(5), 4d(5) ... 
# Wait, cc-pVTZ is 14 AOs: 4s + 3*3p + 2*5d + 1*7f = 4 + 9 + 10 + 7 = 30? No.
# Let's check mol_p.nao
print(f"NAO: {mol_p.nao}")

for i in range(mol_p.nao):
    for j in range(mol_p.nao):
        if abs(s1_p[i,j] - 1.0) < 1e-6: continue
        # Find which index in f matches p[i,:]
        pass

# Let's just print the first 14x14 of S_f vs S_p
np.set_printoptions(precision=3, suppress=True, linewidth=100)
print("PySCF S (first atom):\n", s1_p)
print("QChem S (first atom):\n", s1_f)
