from pyscf import gto, dft

import esipy

mol = gto.Mole()
mol.atom = '''
C  0.00000000  0.00000000  2.63256991
C  0.00000000  2.27987331  1.31628495
C  0.00000000  2.27987331 -1.31628495
C  0.00000000  0.00000000 -2.63256991
C  0.00000000 -2.27987331 -1.31628495
C  0.00000000 -2.27987331  1.31628495
H  0.00000000  0.00000000  4.69242996
H  0.00000000  4.06376154  2.34622537
H  0.00000000  4.06376154 -2.34622537
H  0.00000000  0.00000000 -4.69242996
H  0.00000000 -4.06376154 -2.34622537
H  0.00000000 -4.06376154  2.34622537
'''
mol.basis = '6-311G**'
mol.spin = 0
mol.charge = 0
mol.symmetry = True
mol.verbose = 0
mol.max_memory = 4000
mol.build()

mf = dft.RKS(mol)
mf.xc = "B3LYP"
mf.kernel()

ring = [1, 2, 3, 4, 5, 6]
name = "example01"

for part in ["m", "lowdin", "meta-lowdin", "nao", "iao"]:
    save = name + '_' + part
    arom = esipy.ESI(mol=mol, mf=mf, rings=ring, partition=part, save=save,
                     mci=True, av1245=True)
    arom.print()
    arom.writeaoms(name)
