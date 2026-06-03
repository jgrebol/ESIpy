import numpy as np
from pyscf import gto
from esipy.readfchk import readfchk, read_list_from_fchk
import os

path = "../tests/FCHK/GAUSSIAN/2_benzene_cartesian.fchk"
mol_f, mf_f = readfchk(path)
S_f = mf_f.get_ovlp()

# PySCF native Benzene Cartesian
basis = 'cc-pVTZ'
atom = """
C        0.000000000      0.000000000      1.393096000
C        0.000000000      1.206457000      0.696548000
C        0.000000000      1.206457000     -0.696548000
C        0.000000000      0.000000000     -1.393096000
C        0.000000000     -1.206457000     -0.696548000
C        0.000000000     -1.206457000      0.696548000
H        0.000000000      0.000000000      2.483127000
H        0.000000000      2.150450000      1.241569000
H        0.000000000      2.150450000     -1.241569000
H        0.000000000      0.000000000     -2.483127000
H        0.000000000     -2.150450000     -1.241569000
H        0.000000000     -2.150450000      1.241569000
"""
mol_p = gto.M(atom=atom, basis=basis, cart=True).build()
S_p = mol_p.intor('int1e_ovlp')

# Let's look at the first D-shell of Carbon 1.
# cc-pVTZ for C has: 4s, 3p, 2d, 1f.
# Cartesian 6D shells.
# s: 0,1,2,3
# p: 4-6, 7-9, 10-12
# d: 13-18 (6 functions), 19-24 (6 functions)
# f: 25-34 (10 functions)

print("Comparing Carbon 1, First 6D shell (indices 13-18):")
s1_p = S_p[13:19, 13:18]
s1_f = S_f[13:19, 13:18]

np.set_printoptions(precision=4, suppress=True)
print("\nPySCF S (6D block):\n", S_p[13:19, 13:19])
print("\nFCHK S (6D block):\n", S_f[13:19, 13:19])

# Check mapping within this 6x6 block
for i in range(6):
    for j in range(6):
        if np.allclose(S_p[13+i, :], S_f[13+j, :], atol=1e-5):
             print(f"  PySCF index {13+i} matches FCHK index {13+j}")
