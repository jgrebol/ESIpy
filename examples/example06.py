import esi
import numpy as np
from pyscf import gto, dft

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

homaref = {
    'CC': {'r_opt': 1.437, 'alpha': 950.74},
    'CN': {'r_opt': 1.390, 'alpha': 506.43},
    'NN': {'r_opt': 1.375, 'alpha': 187.36},
    'CO': {'r_opt': 1.379, 'alpha': 164.96},
}

connec = ['C', 'C','C','C','C','C']
ring = [7,3,1,2,6,10]
partition = 'nao'
geom = mol.atom_coords()

esi.aromaticity('example01_nao.aoms', rings=ring, partition=partition, mci=True, av1245=True, homarefs=homaref, connectivity=connec, geom=geom)
