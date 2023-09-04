import esi 
from pyscf import gto, dft

molname = 'benzene'

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
mol.symmetry = False
mol.verbose = 0
mol.max_memory = 4000
mol.build()

mf = dft.RKS(mol)
mf.xc = 'B3LYPg'
mf.kernel()

ring = [7,3,1,2,6,10]
calc = 'meta_lowdin'

Smo = esi.make_aom(mol,mf,calc=calc)
esi.aromaticity(mol, mf, Smo, ring, calc=calc, mci=True, av1245=True, num_threads=1)

with open('name.npy', 'rb') as f:
    Smo = np.load(f) # Loading the AOMs from a binary file

