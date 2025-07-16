from pyscf import gto, dft

import esipy

mol = gto.Mole()
mol.atom = '''
C       -2.989895238      0.000000000      0.822443952
C       -2.989895238      1.206457000      0.125895952
C       -2.989895238      1.206457000     -1.267200048
C       -2.989895238      0.000000000     -1.963748048
C       -2.989895238     -1.206457000     -1.267200048
C       -2.989895238     -1.206457000      0.125895952
H       -2.989895238      0.000000000      1.912474952
H       -2.989895238      2.150450000      0.670916952
H       -2.989895238      2.150450000     -1.812221048
H       -2.989895238      0.000000000     -3.053779048
H       -2.989895238     -2.150450000     -1.812221048
H       -2.989895238     -2.150450000      0.670916952
'''
mol.basis = 'sto-3g'
mol.spin = 0
mol.charge = 0
mol.symmetry = True
mol.verbose = 0
mol.max_memory = 4000
mol.build()

mf = dft.RKS(mol)
mf.xc = "B3LYP"
mf.kernel()

ring = [1, 2, 3, 4, 5, 6]
name = "example01"

for part in ["mulliken", "lowdin", "meta_lowdin", "nao", "iao"]:
    arom = esipy.ESI(mol=mol, mf=mf, rings=ring, partition=part, save=name, mci=True, av1245=True) # mci and av1245 are optional arguments
    arom.print()
    arom.writeaoms(name) # Will take the form "name_partition/"
