import esipy

partition = ['m', 'lowdin', 'meta_lowdin', 'nao', 'iao']
ring = [1, 2, 3, 4, 5, 6]

for part in partition:
    aom = 'example01_' + part + '.aoms'
    molinfo = 'example01_' + part + '.molinfo'
    arom = esipy.ESI(aom=aom, rings=ring, partition=part, molinfo=molinfo)
    arom.print()
