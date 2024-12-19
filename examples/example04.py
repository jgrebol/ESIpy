import esipy

ring = [1, 2, 3, 4, 5, 6]
partition = 'nao'

fluref = {'CC': 1.500}
connectivity = ['C', 'C', 'C', 'C', 'C', 'C']
Smo = 'example01_nao.aoms'
molinfo = 'example01_nao.molinfo'

arom = esipy.ESI(Smo=Smo, molinfo=molinfo, rings=ring, partition=partition, flurefs=fluref, connectivity=connectivity)
arom.print()
