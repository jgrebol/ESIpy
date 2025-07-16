from pyscf import gto, dft

from esipy import ESI

molname = 'example06_restricted'

mol = gto.Mole()
mol.atom = '''
C       -2.989895238      0.000000000      0.822443952
C       -2.989895238      1.206457000      0.125895952
C       -2.989895238      1.206457000     -1.267200048
C       -2.989895238      0.000000000     -1.963748048
C       -2.989895238     -1.206457000     -1.267200048
C       -2.989895238     -1.206457000      0.125895952
H       -2.989895238      0.000000000      1.912474952
H       -2.989895238      2.150450000      0.670916952
H       -2.989895238      2.150450000     -1.812221048
H       -2.989895238      0.000000000     -3.053779048
H       -2.989895238     -2.150450000     -1.812221048
H       -2.989895238     -2.150450000      0.670916952
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

ring = [1, 2, 3, 4, 5, 6]
partition = 'nao'

esi = ESI(mol=mol, mf=mf, partition=partition, rings=ring)

print('Doing for restricted calculations')

iring = esi.indicators[0].iring
print('The Iring is', iring)

mci = esi.indicators[0].mci
print('The MCI using 1 core is', mci)

esi.ncores = 2
mci = esi.indicators[0].mci
print('The MCI using 2 cores is', mci)

av1245 = esi.indicators[0].av1245
print('The AV1245 is', av1245)

avmin = esi.indicators[0].avmin
print('The AVmin is', avmin)

pdi = esi.indicators[0].pdi
print('The PDI is', pdi)

flu = esi.indicators[0].flu
print('The FLU is', flu)

boa = esi.indicators[0].boa
print('The BOA is', boa)

boa_c = esi.indicators[0].boa_c
print('The BOA_c is', boa_c)

homa = esi.indicators[0].homa
print('The HOMA is', homa)

en = esi.indicators[0].en
print('The EN component is', en)

geo = esi.indicators[0].geo
print('The GEO component is', geo)

bla = esi.indicators[0].bla
print('The BLA is', bla)

bla_c = esi.indicators[0].bla_c
print('The BLA_c is', bla_c)

molname = 'example06_unrestricted'

mol = gto.Mole()
mol.atom = '''
C       -2.989895238      0.000000000      0.822443952
C       -2.989895238      1.206457000      0.125895952
C       -2.989895238      1.206457000     -1.267200048
C       -2.989895238      0.000000000     -1.963748048
C       -2.989895238     -1.206457000     -1.267200048
C       -2.989895238     -1.206457000      0.125895952
H       -2.989895238      0.000000000      1.912474952
H       -2.989895238      2.150450000      0.670916952
H       -2.989895238      2.150450000     -1.812221048
H       -2.989895238      0.000000000     -3.053779048
H       -2.989895238     -2.150450000     -1.812221048
H       -2.989895238     -2.150450000      0.670916952
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

iring_alpha = esi.indicators[0].iring_alpha
iring_beta = esi.indicators[0].iring_beta
iring = esi.indicators[0].iring
print('The alpha component of the Iring is', iring_alpha)
print('The beta component of the Iring is', iring_beta)
print('The Iring is', iring)

mci_alpha = esi.indicators[0].mci_alpha
mci_beta = esi.indicators[0].mci_beta
mci = esi.indicators[0].mci
print('The alpha component of the MCI is', mci_alpha)
print('The beta component of the MCI is', mci_beta)
print('The MCI using 1 core is', mci)

esi.ncores = 2
mci_alpha = esi.indicators[0].mci_alpha
mci_beta = esi.indicators[0].mci_beta
mci = esi.indicators[0].mci
print('The alpha component of the MCI is', mci_alpha)
print('The beta component of the MCI is', mci_beta)
print('The MCI using 2 cores is', mci)

av1245_alpha = esi.indicators[0].av1245_alpha
av1245_beta = esi.indicators[0].av1245_beta
av1245 = esi.indicators[0].av1245
print('The alpha component of the AV1245 is', av1245_alpha)
print('The beta component of the AV1245 is', av1245_beta)
print('The AV1245 is', av1245)

avmin_alpha = esi.indicators[0].avmin_alpha
avmin_beta = esi.indicators[0].avmin_beta
avmin = esi.indicators[0].avmin
print('The alpha component of the AVmin is', avmin_alpha)
print('The beta component of the AVmin is', avmin_beta)
print('The AVmin is', avmin)

pdi_alpha = esi.indicators[0].pdi_alpha
pdi_beta = esi.indicators[0].pdi_beta
pdi = esi.indicators[0].pdi
print('The alpha component of the PDI is', pdi_alpha)
print('The beta component of the PDI is', pdi_beta)
print('The PDI is', pdi)

flu_alpha = esi.indicators[0].flu_alpha
flu_beta = esi.indicators[0].flu_beta
flu = esi.indicators[0].flu
print('The alpha component of the FLU is', flu_alpha)
print('The beta component of the FLU is', flu_beta)
print('The FLU is', flu)

boa_alpha = esi.indicators[0].boa_alpha
boa_beta = esi.indicators[0].boa_beta
boa = esi.indicators[0].boa
print('The alpha component of the BOA is', boa_alpha)
print('The beta component of the BOA is', boa_beta)
print('The BOA is', boa)

boa_c_alpha = esi.indicators[0].boa_c_alpha
boa_c_beta = esi.indicators[0].boa_c_beta
boa_c = esi.indicators[0].boa_c
print('The alpha component of the BOA_c is', boa_c_alpha)
print('The beta component of the BOA_c is', boa_c_beta)
print('The BOA_c is', boa_c)
