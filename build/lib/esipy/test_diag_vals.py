import numpy as np
from esipy.readfchk import readfchk
from pyscf import gto

path = "../tests/FCHK/GAUSSIAN/2_benzene_cartesian.fchk"
mol_f, mf_f = readfchk(path)
s_p = mol_f.pyscf_mol.intor('int1e_ovlp')
diag_s = np.diag(s_p)

# Find D shells
shell_types = mol_f.fchk.mssh
cursor = 0
for i, st in enumerate(shell_types):
    if st == 2:
        print(f"Shell {i} (6D) Diag:", diag_s[cursor:cursor+6])
        cursor += 6
    elif st == 3:
        print(f"Shell {i} (10F) Diag:", diag_s[cursor:cursor+10])
        cursor += 10
    elif st == 1: cursor += 3
    elif st == 0: cursor += 1
    elif st == -1: cursor += 4
