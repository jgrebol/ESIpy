import esi
import numpy as np
from pyscf import gto, dft

molname = 'example04'

mol=gto.Mole()
mol.atom='''
6        1.719993497      0.000000000     -0.797854839
6        1.719993497      1.207953000     -1.560011839
6        1.719993497     -1.207953000     -1.560011839
1        1.719993497      2.173678000     -1.051053839
1        1.719993497     -2.173678000     -1.051053839
6        1.719993497      1.207953000     -2.989129839
6        1.719993497     -1.207953000     -2.989129839
1        1.719993497      2.173678000     -3.498087839
1        1.719993497     -2.173678000     -3.498087839
6        1.719993497      0.000000000     -3.751286839
1        1.719993497      0.000000000     -4.836189839
1        1.719993497      0.000000000      0.287048161
'''
mol.basis = 'sto-3g'
mol.spin = 2
mol.charge = 0
mol.symmetry = True
mol.verbose = 0
mol.max_memory = 4000
mol.build()

mf = dft.UKS(mol)
mf.xc = 'B3LYP'
mf.kernel()

ring = [7,3,1,2,6,10]
partition = ['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao']

for part in partition:
    Smo = esi.make_aoms(mol, mf, partition=part, save=molname + '_' + part + '.aoms')
    molinfo = esi.mol_info(mol, mf, partition=part, save=molname + '_' + part +  '.molinfo')
    esi.aromaticity(Smo, rings=ring, mol=mol, mf=mf, partition=part, mci=True, av1245=True, num_threads=1)
