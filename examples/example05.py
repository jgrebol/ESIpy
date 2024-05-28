import esi
import numpy as np
from pyscf import gto, dft

ring = [7,3,1,2,6,10]
partition = 'nao'

fluref = {'CC': 1.500}
connec = ['C', 'C','C','C','C','C']

esi.aromaticity('example01_nao.aoms', rings=ring, partition=partition, mci=True, av1245=True, flurefs=fluref, connectivity=connec)
