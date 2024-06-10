import esi

partition = ['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao']

for part in partition:
    Smo = 'example01_' + part + '.aoms'
    molinfo = 'example01_' + part + '.molinfo'
    esi.aromaticity(Smo, rings=[7,3,1,2,6,10], molinfo=molinfo, mci=True, av1245=True, partition=part)
