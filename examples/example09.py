import esipy

ring = [1, 2, 3, 4, 5, 6]
partition = 'metalow'
path = '../joan/bzqtaim_atomicfiles'
#path = '../joan/bz_metalow'

# By default, will search for ".int" in the working directory
arom = esipy.ESI(read=True, readpath=path, partition=partition, rings=ring, mci=True, av1245=True)
arom.readaoms()
# Now the variable arom.aom contains the AOMs in the ./example08_nao/ directory
arom.print()
