import esi
from pyscf import gto

mol=gto.Mole()
mol.atom='''
6       -1.366007715     -0.671871000     -3.014007296
6       -2.636254715     -1.421829000     -3.014007296
6       -3.906495715     -0.671871000     -3.014007296
6       -3.906495715      0.671871000     -3.014007296
6       -2.636248715      1.421829000     -3.014007296
6       -1.366007715      0.671871000     -3.014007296
1       -0.428150715     -1.228590000     -3.014007296
1       -2.636254715     -2.509051000     -3.014007296
1       -4.844354715     -1.228588000     -3.014007296
1       -4.844352715      1.228590000     -3.014007296
1       -2.636248715      2.509051000     -3.014007296
1       -0.428148715      1.228588000     -3.014007296
'''
mol.basis = 'sto-3g'
mol.spin = 2
mol.charge = 0
mol.symmetry = True
mol.verbose = 0
mol.max_memory = 4000
mol.build()

homerref = {'CC': {'r_opt': 1.388, 'alpha': 257.7}}

connec = ['C', 'C','C','C','C','C']
ring = [1,2,3,4,5,6]
partition = 'nao'
geom = mol.atom_coords()

esi.aromaticity('example03_nao.aoms', rings=ring, partition=partition, mci=True, av1245=True, homerrefs=homerref, connectivity=connec, geom=geom)


