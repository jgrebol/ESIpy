import esi 
import numpy as np

ring = [7,3,1,2,6,10]
calc = 'meta_lowdin'
wf = 'rest'

with open('benzene.npy', 'rb') as f:
    Smo = np.load(f) # Loading the AOMs from a binary file

esi.aromaticity_from_aoms(Smo, ring, calc=calc, wf=wf, mci=True, av1245=True, num_threads=1)

wf = 'unrest'

with open('benzene_unrest.npy', 'rb') as f:
   Smo = np.load(f)

esi.aromaticity_from_aoms(Smo, ring, calc=calc, wf=wf, mci=True, av1245=True, num_threads=1)

