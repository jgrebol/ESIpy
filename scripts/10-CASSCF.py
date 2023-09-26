import esi 
from pyscf import gto, scf, lib, mcscf
import numpy as np

molname = 'benzene_casscf'

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
#mol.cart= True
mol.symmetry = True
mol.verbose = 0
mol.max_memory = 4000
mol.build()

mf = scf.RHF(mol)
mf.kernel()

mc = mcscf.CASSCF(mf, 4, 4)
mc.verbose = 0
mc.kernel()

ring = [7,3,1,2,6,10]
calc = 'mulliken'

Smo = esi.make_aom(mol,mc,calc=calc)
esi.aromaticity(mol, mc, Smo, ring, calc=calc, mci=True, av1245=True, num_threads=2)


