from pyscf import gto, scf, ci, cc, mp, mcscf

import esipy

molname = 'benzene'

mol = gto.Mole()
mol.atom = '''
C       -2.250458781      0.000000000     -0.958601895
C       -2.250458781      1.207953000     -1.720758895
C       -2.250458781      1.207953000     -3.149876895
C       -2.250458781      0.000000000     -3.912033895
C       -2.250458781     -1.207953000     -3.149876895
C       -2.250458781     -1.207953000     -1.720758895
H       -2.250458781      2.173678000     -1.211800895
H       -2.250458781     -2.173678000     -1.211800895
H       -2.250458781      2.173678000     -3.658834895
H       -2.250458781     -2.173678000     -3.658834895
H       -2.250458781      0.000000000     -4.996936895
H       -2.250458781      0.000000000      0.126301105
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

for part in ["mulliken", "lowdin", "meta-lowdin", "nao", "iao"]:
    for method in [mf1, mf2, mf3, mf4]:
        arom = esipy.ESI(mol=mol, mf=method, myhf=mf, rings=ring, partition=part, ncores=1)
        arom.print()
