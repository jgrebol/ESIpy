import esipy
from esipy.atomicfiles import write_aoms
from esipy.make_aoms import make_aoms
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

mf = dft.RKS(mol)
mf.xc = 'B3LYPg'
mf.kernel()

ring = [7,3,1,2,6,10]
partition = 'nao'
name = 'example08'
molinfo_name = name + '_' + partition + '.molinfo'
aoms_name = name + '_' + partition + '.aoms'

arom = esipy.ESI(rings=ring, partition=partition, mol=mol, mf=mf, name=name, saveaoms=aoms_name, savemolinfo=molinfo_name)
arom.print()
arom.writeaoms()
