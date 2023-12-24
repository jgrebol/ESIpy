import esi
from pyscf import gto, scf

molname = 'test'

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
mol.symmetry = False
mol.verbose = 0
mol.max_memory = 4000
mol.build()

mf = scf.RHF(mol)
mf.kernel()

ring = [7,3,1,2,6,10]
calc = 'meta_lowdin'

Smo = esi.make_aoms(mol,mf,calc=calc)
esi.write_int(mol, mf, molname, Smo, ring, calc=calc)


