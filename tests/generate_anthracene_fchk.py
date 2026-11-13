"""
Generate 13_anthracene.fchk for ESIpy tests using APOST3D's write_fchk.
Anthracene, RHF/sto-3g.  Run once; commit the resulting FCHK file.
"""
import sys
sys.path.insert(0, '/home/joan/APOST3D/utils')

from pyscf import gto, scf
from apost3d import write_fchk

mol = gto.Mole()
mol.atom = """
C   0.   2.4834  1.409
C   0.   3.6664  0.7142
C   0.   3.6664 -0.7142
C   0.   2.4834 -1.409
C   0.  -2.4834 -1.409
C   0.  -3.6664 -0.7142
C   0.  -3.6664  0.7142
C   0.  -2.4834  1.409
C   0.   0.      1.4056
C   0.   0.     -1.4056
C   0.   1.2258  0.7233
C   0.   1.2258 -0.7233
C   0.  -1.2258 -0.7233
C   0.  -1.2258  0.7233
H   0.   2.4815  2.5021
H   0.   4.6178  1.2509
H   0.   4.6178 -1.2509
H   0.   2.4815 -2.5021
H   0.  -2.4815 -2.5021
H   0.  -4.6178 -1.2509
H   0.  -4.6178  1.2509
H   0.  -2.4815  2.5021
H   0.   0.      2.4994
H   0.   0.     -2.4994
"""
mol.basis = 'sto-3g'
mol.unit = 'angstrom'
mol.spin = 0
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mf.kernel()

S = mol.intor('int1e_ovlp')
outpath = 'tests/FCHK/GAUSSIAN/13_anthracene'
write_fchk(mol, mf, outpath, S)
print(f"\nWrote {outpath}.fchk")
