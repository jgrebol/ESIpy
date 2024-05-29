import esi
from pyscf import gto, dft

molname = 'example08'

mol=gto.Mole()
mol.atom='''
6       -2.668401126      1.246700000     -3.249227265
6       -2.668401126      2.437400000     -3.944227265
6       -2.668401126      2.437400000     -5.363027265
6       -2.668401126      1.246700000     -6.058027265
6       -2.668401126     -1.246700000     -6.058027265
6       -2.668401126     -2.437400000     -5.363027265
6       -2.668401126     -2.437400000     -3.944227265
6       -2.668401126     -1.246700000     -3.249227265
6       -2.668401126      0.000000000     -3.936027265
6       -2.668401126      0.000000000     -5.371227265
1       -2.668401126      1.244200000     -2.155927265
1       -2.668401126      3.387000000     -3.404127265
1       -2.668401126      3.387000000     -5.903127265
1       -2.668401126      1.244200000     -7.151327265
1       -2.668401126     -1.244200000     -7.151327265
1       -2.668401126     -3.387000000     -5.903127265
1       -2.668401126     -3.387000000     -3.404127265
1       -2.668401126     -1.244200000     -2.155927265
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

ring = [[1,2,3,4,10,9],[5,6,7,8,9,10],[1,2,3,4,10,5,6,7,8,9]]
partition = ['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao']

for part in partition:
    Smo = esi.make_aoms(mol, mf, partition=part, save=molname + '_' + part + '.aoms')
    molinfo = esi.mol_info(mol, mf, partition=part, save=molname + '_' + part +  '.molinfo')
    esi.aromaticity(Smo, rings=ring, mol=mol, mf=mf, partition=part, mci=True, av1245=True, num_threads=2)
