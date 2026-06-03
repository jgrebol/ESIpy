import numpy as np
from esipy.readfchk import readfchk
from pyscf import gto, scf
import os

path = "../tests/FCHK/GAUSSIAN/1_benzene_spherical.fchk"
mol_f, mf_f = readfchk(path)
dm_f = mf_f.make_rdm1()

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
mol_p = gto.M(atom=atom, basis='cc-pVTZ').build()
S_p = mol_p.intor('int1e_ovlp')

print(f"Trace(P_fchk @ S_pyscf) with NEW SDIAG: {np.trace(dm_f @ S_p):.4f}")
