from pyscf import gto, dft

from esipy import ESI

molname = 'example06_restricted'

mol = gto.Mole()
mol.atom = '''
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

ring = [7, 3, 1, 2, 6, 10]
partition = 'nao'

esi = ESI(mol=mol, mf=mf, partition=partition, rings=ring)

# For restricted calculations - add the 2* factor for the doubly occupied MOs
print('Doing for restricted calculations')

iring = esi.indicators.iring
print('The Iring is', iring)

mci = esi.indicators.mci
print('The MCI using 1 core is', mci)

esi.ncores = 2
mci = esi.indicators.mci
print('The MCI using 2 cores is', mci)

av1245 = esi.indicators.av1245
print('The AV1245 is', av1245)

avmin = esi.indicators.avmin
print('The AVmin is', avmin)

pdi = esi.indicators.pdi
print('The PDI is', pdi)

flu = esi.indicators.flu
print('The FLU is', flu)

boa = esi.indicators.boa
print('The BOA is', boa)

boa_c = esi.indicators.boa_c
print('The BOA_c is', boa_c)

homa = esi.indicators.homa
print('The HOMA is', homa)

en = esi.indicators.en
print('The EN component is', en)

geo = esi.indicators.geo
print('The GEO component is', geo)

bla = esi.indicators.bla
print('The BLA is', bla)

bla_c = esi.indicators.bla_c
print('The BLA_c is', bla_c)

molname = 'example06_unrestricted'

mol = gto.Mole()
mol.atom = '''
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

esi = ESI(mol=mol, mf=mf, partition=partition, rings=ring)

print('Doing for unrestricted calculation')

iring_alpha = esi.indicators.iring_alpha
iring_beta = esi.indicators.iring_beta
iring = esi.indicators.iring
print('The alpha component of the Iring is', iring_alpha)
print('The beta component of the Iring is', iring_beta)
print('The Iring is', iring)

mci_alpha = esi.indicators.mci_alpha
mci_beta = esi.indicators.mci_beta
mci = esi.indicators.mci
print('The alpha component of the MCI is', mci_alpha)
print('The beta component of the MCI is', mci_beta)
print('The MCI using 1 core is', mci)

esi.ncores = 2
mci_alpha = esi.indicators.mci_alpha
mci_beta = esi.indicators.mci_beta
mci = esi.indicators.mci
print('The alpha component of the MCI is', mci_alpha)
print('The beta component of the MCI is', mci_beta)
print('The MCI using 2 cores is', mci)

av1245_alpha = esi.indicators.av1245_alpha
av1245_beta = esi.indicators.av1245_beta
av1245 = esi.indicators.av1245
print('The alpha component of the AV1245 is', av1245_alpha)
print('The beta component of the AV1245 is', av1245_beta)
print('The AV1245 is', av1245)

avmin_alpha = esi.indicators.avmin_alpha
avmin_beta = esi.indicators.avmin_beta
avmin = esi.indicators.avmin
print('The alpha component of the AVmin is', avmin_alpha)
print('The beta component of the AVmin is', avmin_beta)
print('The AVmin is', avmin)

pdi_alpha = esi.indicators.pdi_alpha
pdi_beta = esi.indicators.pdi_beta
pdi = esi.indicators.pdi
print('The alpha component of the PDI is', pdi_alpha)
print('The beta component of the PDI is', pdi_beta)
print('The PDI is', pdi)

flu_alpha = esi.indicators.flu_alpha
flu_beta = esi.indicators.flu_beta
flu = esi.indicators.flu
print('The alpha component of the FLU is', flu_alpha)
print('The beta component of the FLU is', flu_beta)
print('The FLU is', flu)

boa_alpha = esi.indicators.boa_alpha
boa_beta = esi.indicators.boa_beta
boa = esi.indicators.boa
print('The alpha component of the BOA is', boa_alpha)
print('The beta component of the BOA is', boa_beta)
print('The BOA is', boa)

boa_c_alpha = esi.indicators.boa_c_alpha
boa_c_beta = esi.indicators.boa_c_beta
boa_c = esi.indicators.boa_c
print('The alpha component of the BOA_c is', boa_c_alpha)
print('The beta component of the BOA_c is', boa_c_beta)
print('The BOA_c is', boa_c)
