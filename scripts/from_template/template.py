import esi 
from pyscf import gto, dft

molname = ''

mol=gto.Mole()
mol.atom='''
'''
mol.basis = 'sto-3g'
mol.spin = 0
mol.charge = 0
#mol.cart= False
mol.symmetry = False
mol.verbose = 0
mol.max_memory = 4000
mol.build()

mf = dft.RKS(mol)
mf.xc = 'B3LYPg'
mf.kernel()

ring = []
calc = ''

Smo = esi.make_aom(mol,mf,calc=calc)
esi.aromaticity(mol, mf, molname, Smo, ring, calc=calc, mci=False, av1245=False, num_threads=1)


