import esipy.indicators as ind
from esipy.rest import deloc_rest
from esipy.unrest import deloc_unrest
from esipy import ESI
from pyscf import gto, dft

molname = 'example06_restricted'

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

mf = dft.RKS(mol)
mf.xc = 'B3LYP'
mf.kernel()

ring = [7,3,1,2,6,10]
partition = 'nao'

esi = ESI(mol=mol, mf=mf, partition=partition)
Smo = esi.Smo

# For restricted calculations - add the 2* factor for the doubly occupied MOs
print('Doing for restricted calculations')

iring = ind.compute_iring(ring, Smo)
print('The Iring is', 2 * iring) 

mci = ind.sequential_mci(ring, Smo, partition=partition)
print('The MCI using 1 core is', 2 * mci) 

mci = ind.multiprocessing_mci(ring, Smo, ncores=2, partition=partition)
print('The MCI using 2 cores is', 2 * mci) 

av1245 = ind.compute_av1245(ring, Smo)[0]
print('The AV1245 is', 2 * av1245)

avmin = ind.compute_av1245(ring, Smo)[1]
print('The AVmin is', 2 * avmin)

pdi = ind.compute_pdi(ring, Smo)[0]
print('The PDI is', 2 * pdi)

flu = ind.compute_flu(ring, mol, Smo, partition=partition)
print('The FLU is', 2 * flu)

boa = ind.compute_boa(ring, Smo)[0]
print('The BOA is', 2 * boa)

boa_c = ind.compute_boa(ring, Smo)[1]
print('The BOA_c is', 2 * boa_c)

homa = ind.compute_homa(ring, mol)[0]
print('The HOMA is',  homa)

bla = ind.compute_bla(ring, mol)[0]
print('The BLA is', bla)

bla_c = ind.compute_bla(ring, mol)[1]
print('The BLA_c is', bla_c)

#To compute only the delocalization indices and atomic populations for restricted calculations
deloc_rest(Smo, mol)

molname = 'example06_unrestricted'

mol=gto.Mole()
mol.atom='''
6        1.719993497      0.000000000     -0.797854839
6        1.719993497      1.207953000     -1.560011839
6        1.719993497     -1.207953000     -1.560011839
1        1.719993497      2.173678000     -1.051053839
1        1.719993497     -2.173678000     -1.051053839
6        1.719993497      1.207953000     -2.989129839
6        1.719993497     -1.207953000     -2.989129839
1        1.719993497      2.173678000     -3.498087839
1        1.719993497     -2.173678000     -3.498087839
6        1.719993497      0.000000000     -3.751286839
1        1.719993497      0.000000000     -4.836189839
1        1.719993497      0.000000000      0.287048161
'''
mol.basis = 'sto-3g'
mol.spin = 2
mol.charge = 0
mol.symmetry = False
mol.verbose = 0
mol.max_memory = 4000
mol.build()

mf = dft.UKS(mol)
mf.xc = 'B3LYP'
mf.kernel()

esi = ESI(mol=mol, mf=mf, partition=partition)
Smo = esi.Smo

print('Doing for unrestricted calculation')

iring_alpha = ind.compute_iring(ring, Smo[0])
iring_beta = ind.compute_iring(ring, Smo[1])
print('The Iring is', iring_alpha + iring_beta) 

mci_alpha = ind.sequential_mci(ring, Smo[0], partition=partition)
mci_beta = ind.sequential_mci(ring, Smo[1], partition=partition)
print('The MCI using 1 core is', mci_alpha + mci_beta) 

mci_alpha = ind.multiprocessing_mci(ring, Smo[0], ncores=2, partition=partition)
mci_beta = ind.multiprocessing_mci(ring, Smo[1], ncores=2, partition=partition)
print('The MCI using 2 cores is', mci_alpha+ mci_beta) 

av1245_alpha = ind.compute_av1245(ring, Smo[0])[0]
av1245_beta = ind.compute_av1245(ring, Smo[1])[0]
print('The AV1245 is', av1245_alpha + av1245_beta)

avmin_alpha = ind.compute_av1245(ring, Smo[0])[1]
avmin_beta = ind.compute_av1245(ring, Smo[1])[1]
print('The AVmin is', avmin_alpha + avmin_beta)

pdi_alpha = ind.compute_pdi(ring, Smo[0])[0]
pdi_beta = ind.compute_pdi(ring, Smo[1])[0]
print('The PDI is', pdi_alpha + pdi_beta)

flu_alpha = ind.compute_flu(ring, mol, Smo[0], partition=partition)
flu_beta = ind.compute_flu(ring, mol, Smo[1], partition=partition)
print('The FLU is', flu_alpha + flu_beta)

boa_alpha = ind.compute_boa(ring, Smo[0])[0]
boa_beta = ind.compute_boa(ring, Smo[1])[0]
print('The BOA is', boa_alpha + boa_beta)

boa_c_alpha = ind.compute_boa(ring, Smo[0])[1]
boa_c_beta = ind.compute_boa(ring, Smo[1])[1]
print('The BOA_c is', boa_c_alpha + boa_c_beta)

homer = ind.compute_homer(ring, mol)
print('The HOMER is',homer)

# The BLA does not change between restricted or unrestricted calculations
bla = ind.compute_bla(ring, mol)[0]
print('The BLA is', bla)

bla_c = ind.compute_bla(ring, mol)[1]
print('The BLA_c is', bla_c)

#To compute only the delocalization indices and atomic populations for unrestricted calculations
deloc_unrest(Smo, mol)
