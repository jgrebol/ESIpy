import esipy

partition = ['mulliken', 'lowdin', 'meta-lowdin', 'nao', 'iao']
ring = [1, 2, 3, 4, 5, 6]

for part in partition:
    path = 'example01_' + part + '_esipy/example01_' + part + '_' + part
    aom = path + '.aoms'
    molinfo = path + '.molinfo'
    arom = esipy.ESI(aom=aom, rings=ring, partition=part, molinfo=molinfo)
    arom.print()
