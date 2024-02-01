import esi 
from pyscf import gto, dft

molname = 'anthracene'

mol=gto.Mole()
mol.atom='''
 C                     0.        2.4834    1.409 
 C                     0.        3.6664    0.7142 
 C                     0.        3.6664   -0.7142 
 C                     0.        2.4834   -1.409 
 C                     0.       -2.4834   -1.409 
 C                     0.       -3.6664   -0.7142 
 C                     0.       -3.6664    0.7142 
 C                     0.       -2.4834    1.409 
 C                     0.        0.        1.4056 
 C                     0.        0.       -1.4056 
 C                     0.        1.2258    0.7233 
 C                     0.        1.2258   -0.7233 
 C                     0.       -1.2258   -0.7233 
 C                     0.       -1.2258    0.7233 
 H                     0.        2.4815    2.5021 
 H                     0.        4.6178    1.2509 
 H                     0.        4.6178   -1.2509 
 H                     0.        2.4815   -2.5021 
 H                     0.       -2.4815   -2.5021 
 H                     0.       -4.6178   -1.2509 
 H                     0.       -4.6178    1.2509 
 H                     0.       -2.4815    2.5021 
 H                     0.        0.        2.4994 
 H                     0.        0.       -2.4994 
'''
mol.basis = 'sto-3g'
mol.spin = 0
mol.charge = 0
mol.symmetry = False
mol.verbose = 0
mol.max_memory = 4000
mol.build()

mf = dft.RKS(mol)
mf.xc = 'B3LYP'
mf.kernel()

ring = [[1,2,3,4,12,11],[9,11,12,10,13,14]]
calc = 'meta_lowdin'

Smo = esi.make_aoms(mol,mf,calc=calc)
esi.aromaticity(mol, mf, Smo, ring, calc=calc, mci=True, av1245=True, num_threads=1)


