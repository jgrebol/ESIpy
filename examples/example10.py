from pyscf import gto, scf, ci, cc, mp, mcscf

import esipy

molname = 'benzene'

mol = gto.Mole()
mol.atom = '''
6        0.000000000      0.000000000      1.393096000
6        0.000000000      1.206457000      0.696548000
6        0.000000000      1.206457000     -0.696548000
6        0.000000000      0.000000000     -1.393096000
6        0.000000000     -1.206457000     -0.696548000
6        0.000000000     -1.206457000      0.696548000
1        0.000000000      0.000000000      2.483127000
1        0.000000000      2.150450000      1.241569000
1        0.000000000      2.150450000     -1.241569000
1        0.000000000      0.000000000     -2.483127000
1        0.000000000     -2.150450000     -1.241569000
1        0.000000000     -2.150450000      1.241569000
'''
mol.basis = 'sto-3g'
mol.spin = 0
mol.charge = 0
mol.symmetry = True
mol.verbose = 0
mol.max_memory = 4000
mol.build()

mf = scf.RHF(mol).run()

print("Running CCSD calculation...")
mf1 = cc.CCSD(mf).run()
print("Running CISD calculation...")
mf2 = ci.CISD(mf).run()
print("Running CASSCF calculation...")
mf3 = mcscf.CASSCF(mf, 6, 6).run()
print("Running MP2 calculation...")
mf4 = mp.MP2(mf).run()
ring = [1, 2, 3, 4, 5, 6]

for part in ["mulliken", "lowdin", "meta_lowdin", "nao", "iao"]:
    for method in [mf1, mf2, mf3, mf4]:
        arom = esipy.ESI(mol=mol, mf=mf3, myhf=mf, rings=ring, partition=part, ncores=1)
        arom.print()
