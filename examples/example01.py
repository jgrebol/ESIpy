import esi
from pyscf import gto, dft

molname = 'example01'

mol=gto.Mole()
mol.atom='''
 C                     0.       -1.39633   0.
 C                    -1.20926  -0.69816   0.
 C                     1.20926  -0.69816   0.
 H                    -2.15006  -1.24134   0.
 H                     2.15006  -1.24134   0.
 C                    -1.20926   0.69816   0.
 C                     1.20926   0.69816   0.
 H                    -2.15006   1.24134   0.
 H                     2.15006   1.24134   0.
 C                     0.        1.39633   0.
 H                     0.        2.48268   0.
 H                     0.       -2.48268   0.
'''
mol.basis = 'sto-3g'
mol.spin = 0
mol.charge = 0
mol.symmetry = True
mol.verbose = 0
mol.max_memory = 4000
mol.build()

mf = dft.RKS(mol)
mf.xc = 'B3LYP'
mf.kernel()

ring = [7,3,1,2,6,10]
partition = ['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao']

for part in partition:
    # Creates a .aoms file for each AIM with the AOMs
    Smo = esi.make_aoms(mol, mf, partition=part, save=molname + '_' + part + '.aoms')
    # Creates an additional -molinfo file for each AIM (optional but recommended)
    molinfo = esi.mol_info(mol, mf, partition=part, save=molname + '_' + part +  '.molinfo')
    # Calculates the aromaticity indicators
    esi.aromaticity(Smo, rings=ring, mol=mol, mf=mf, partition=part, mci=True, av1245=True, num_threads=1)
