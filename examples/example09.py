import esipy
from esipy.aromaticity import aromaticity
from esipy.atomicfiles import read_aoms

ring = [7,3,1,2,6,10]
partition = 'nao'
molinfo = 'example08_nao.molinfo'
path='example08_nao/'

# By default, will search for ".int" in the working directory
arom = esipy.ESI(readpath=path, molinfo=molinfo, partition=partition, rings=ring, mci=True, av1245=True)
arom.readaoms()

# Now the variable arom.Smo contains the AOMs in the ./example08_nao/ directory
arom.calc()
