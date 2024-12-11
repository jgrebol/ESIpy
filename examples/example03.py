import esipy
from pyscf import gto, dft

mol=gto.Mole()
mol.atom='''
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
mol.spin = 2
mol.charge = 0
mol.symmetry = True
mol.verbose = 0
mol.max_memory = 4000
mol.build()

mf = dft.UKS(mol)
mf.xc = 'B3LYP'
mf.kernel()

ring = [1,2,3,4,5,6]
name = "example03"
for part in ["mulliken", "lowdin", "meta_lowdin", "nao", "iao"]:
    aoms_name = name + '_' + part + '.aoms'
    molinfo_name = name + '_' + part + '.molinfo'
    arom = esipy.ESI(rings=ring, partition=part, mol=mol, mf=mf, saveaoms=aoms_name, savemolinfo=molinfo_name, name=name)
    arom.print()
    arom.writeaoms()
