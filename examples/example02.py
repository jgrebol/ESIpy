import esipy

partition = ['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao']
ring = [1,2,3,4,5,6]

for part in partition:
    Smo = 'example01_' + part + '.aoms'
    molinfo = 'example01_' + part + '.molinfo'
    esipy.ESI(Smo=Smo, rings=ring, partition=part, molinfo=molinfo).calc()
