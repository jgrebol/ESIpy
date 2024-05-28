import esi
import numpy as np
from pyscf import gto, dft

partition = ['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao']

for part in partition:
    esi.aromaticity('example01_' + part + '.aoms', rings=[7,3,1,2,6,10], molinfo='example01_' + part + '.molinfo', mci=True, av1245=True, partition=part)
