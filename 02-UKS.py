import esi 
from pyscf import gto, dft

molname = 'naphthalene'

mol=gto.Mole()
mol.atom='''
C                     0.        1.2467    1.4044
C                     0.        2.4374    0.7094
C                     0.        2.4374   -0.7094
C                     0.        1.2467   -1.4044
C                     0.       -1.2467   -1.4044
C                     0.       -2.4374   -0.7094
C                     0.       -2.4374    0.7094
C                     0.       -1.2467    1.4044
C                     0.        0.        0.7176
C                     0.        0.       -0.7176
H                     0.        1.2442    2.4977
H                     0.        3.387     1.2495
H                     0.        3.387    -1.2495
H                     0.        1.2442   -2.4977
H                     0.       -1.2442   -2.4977
H                     0.       -3.387    -1.2495
H                     0.       -3.387     1.2495
H                     0.       -1.2442    2.4977
'''
mol.basis = 'sto-3g'
mol.spin = 0
mol.charge = 0
mol.symmetry = False
mol.verbose = 0
mol.max_memory = 4000
mol.build()

mf = dft.UKS(mol)
mf.xc = 'B3LYP'
mf.kernel()

ring = [[1,2,3,4,10,9],[,5,6,7,8,9,10]]
calc = 'meta_lowdin'

Smo = esi.make_aoms(mol,mf,calc=calc)
esi.aromaticity(mol, mf, Smo, ring, calc=calc, mci=True, av1245=True, num_threads=1)


