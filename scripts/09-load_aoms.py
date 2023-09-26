import esi 

ring = [7,3,1,2,6,10]
calc = 'meta_lowdin'

# Loading the AOMs from a binary file
with open('name.npy', 'rb') as f:
    Smo = np.load(f) 
 
Smo = esi.make_aom(mol,mf,calc=calc)
esi.aromaticity(mol, mf, Smo, ring, calc=calc, mci=True, av1245=True, num_threads=1)



