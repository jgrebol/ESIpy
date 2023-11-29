import esi 
from pyscf import gto, scf

molname = 'benzene'

mol=gto.Mole()
mol.atom='''
 C                     0.       -1.39633   0.
 C                    -1.20926  -0.69816   0.
 C                     1.20926  -0.69816   0.
 H                    -2.15006  -1.24134   0.
 H                     2.15006  -1.24134   0.
 C                    -1.20926   0.69816   0.
 C                     1.20926   0.69816   0.
 H                    -2.15006   1.24134   0.
 H                     2.15006   1.24134   0.
 C                     0.        1.39633   0.
 H                     0.        2.48268   0.
 H                     0.       -2.48268   0.
'''
mol.basis = 'sto-3g'
mol.spin = 0
mol.charge = 0
mol.symmetry = False
mol.verbose = 0
mol.max_memory = 4000
mol.build()

mf = scf.RHF(mol)
mf.kernel()

ring = [7,3,1,2,6,10]
calc = 'nao'

Smo = esi.make_aoms(mol,mf,calc=calc)

# For unrestricted calculations - add the 2* factor for the doubly occupied MOs
print('Doing for restricted calculations')

iring = esi.compute_iring(ring, Smo)
print('The Iring is', 2 * iring) 

mci = esi.sequential_mci(ring, Smo)
print('The MCI is', 2 * mci) 

av1245 = esi.compute_av1245(ring, Smo)[0]
print('The AV1245 is', 2 * av1245)

pdi = esi.compute_pdi(ring, Smo)[0]
print('The PDI is', 2 * pdi)


molname = 'benzene'

mol=gto.Mole()
mol.atom='''
 C                     0.       -1.39633   0.
 C                    -1.20926  -0.69816   0.
 C                     1.20926  -0.69816   0.
 H                    -2.15006  -1.24134   0.
 H                     2.15006  -1.24134   0.
 C                    -1.20926   0.69816   0.
 C                     1.20926   0.69816   0.
 H                    -2.15006   1.24134   0.
 H                     2.15006   1.24134   0.
 C                     0.        1.39633   0.
 H                     0.        2.48268   0.
 H                     0.       -2.48268   0.
'''
mol.basis = 'sto-3g'
mol.spin = 0
mol.charge = 0
mol.symmetry = False
mol.verbose = 0
mol.max_memory = 4000
mol.build()

mf = scf.UHF(mol)
mf.kernel()

ring = [7,3,1,2,6,10]
calc = 'nao'

Smo = esi.make_aoms(mol,mf,calc=calc)

# For unrestricted calculations
print('Doing for unrestricted calculations')

iring_alpha = esi.compute_iring(ring, Smo[0])
iring_beta = esi.compute_iring(ring, Smo[1])
print('The Iring is', iring_alpha + iring_beta)

mci_alpha = esi.sequential_mci(ring, Smo[0])
mci_beta = esi.sequential_mci(ring, Smo[1])
print('The MCI is', mci_alpha + mci_beta)

av1245_alpha = esi.compute_av1245(ring, Smo[0])[0]
av1245_beta = esi.compute_av1245(ring, Smo[1])[0]
print('The AV1245 is', av1245_alpha + av1245_beta)

pdi_alpha = esi.compute_pdi(ring, Smo[0])[0]
pdi_beta = esi.compute_pdi(ring, Smo[1])[0]
print('The PDI is', pdi_alpha + pdi_beta)



