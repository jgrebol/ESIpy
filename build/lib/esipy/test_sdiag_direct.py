import numpy as np
from esipy.readfchk import readfchk
from pyscf import gto

path = "../tests/FCHK/GAUSSIAN/2_benzene_cartesian.fchk"
mol_f, mf_f = readfchk(path)

# Manual Sdiag application
nao = mol_f.numao
sdiag = np.ones(nao)

SQRT3 = np.sqrt(3.0)
SQRT5 = np.sqrt(5.0)
SQRT7 = np.sqrt(7.0)
SQRT9 = np.sqrt(9.0)

# We need to map which AOs are D, F, G, H
from esipy.tools import permute_aos_rows

# Re-read basics to find shell types
shell_types = mol_f.fchk.mssh
iatsh = mol_f.fchk.iatsh

cursor = 0
for st in shell_types:
    if st == 2: # 6D
        # XX, YY, ZZ, XY, XZ, YZ
        # Sdiag_d = [1.0, 1.0, 1.0, SQRT3, SQRT3, SQRT3] ? 
        # Actually the Fortran order is XX, YY, ZZ, XY, XZ, YZ
        # PySCF order is XX, XY, XZ, YY, YZ, ZZ
        # The Fortran loop does: coeff(i,:) = coeff2(order(i),:)/Sdiag_d(i)
        # order = [1, 4, 5, 2, 6, 3]
        # Sdiag_d(1)=1.0 (XX), Sdiag_d(2)=1.0 (YY), Sdiag_d(3)=1.0 (ZZ), 
        # Sdiag_d(4)=SQRT3 (XY), Sdiag_d(5)=SQRT3 (XZ), Sdiag_d(6)=SQRT3 (YZ)
        sdiag[cursor:cursor+6] = [1.0, 1.0, 1.0, SQRT3, SQRT3, SQRT3]
        cursor += 6
    elif st == 3: # 10F
        # XXX, YYY, ZZZ, XYY, XXY, XXZ, XZZ, YZZ, YYZ, XYZ
        sdiag[cursor:cursor+10] = [1.0, 1.0, 1.0, SQRT3, SQRT3, SQRT3, SQRT3, SQRT3, SQRT3, SQRT3*SQRT5] # Placeholder
        # Actually XXX, YYY, ZZZ are 1.0. XYY, XXY etc are SQRT3. XYZ is SQRT15.
        sdiag[cursor:cursor+10] = [1.0, 1.0, 1.0, SQRT3, SQRT3, SQRT3, SQRT3, SQRT3, SQRT3, SQRT3*SQRT5]
        cursor += 10
    elif st == 1: cursor += 3
    elif st == 0: cursor += 1
    elif st == -1: cursor += 4
    else:
        # High L...
        l = st
        if l == 4: n=15
        elif l == 5: n=21
        else: n = (l+1)*(l+2)//2
        cursor += n

# Now we need to reach the RAW coefficients before standardize_mat
# Or just UN-SCALE what readfchk did and RE-SCALE correctly.
# readfchk did: return mat_p * v[:, None] where v = 1/sqrt(diag(S_raw))
# We want: return mat_p / sdiag_reordered

v_applied = 1.0 / np.sqrt(np.abs(np.diag(mol_f.pyscf_mol.intor('int1e_ovlp'))))
# Note: readfchk.py: MeanField2.__init__ calculates S_raw and v.
# Then standardize_mat applies it.
# BUT it applies it AFTER permutation.

# Let's just modify readfchk.py to use fixed Sdiag if it's Gaussian.
