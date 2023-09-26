import esi 
from pyscf import gto, dft, scf # In this case scf is mandatory

molname = '08-from_chk'

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

#1 To save the checkout file as a .pychk
mf.chkfile = molname + '.pychk'
mf.kernel()

#2 Starting the kernel from the density matrix stored in the .pychk file, performing only one cicle
mo_coeff = scf.chkfile.load(molname+'.pychk', 'scf/mo_coeff')
mo_occ = scf.chkfile.load(molname+'.pychk', 'scf/mo_occ')
dm = mf.make_rdm1(mo_coeff, mo_occ)
mf.kernel(dm)

ring = [7,3,1,2,6,10]
calc = 'meta_lowdin'

Smo = esi.make_aom(mol,mf,calc=calc)
esi.aromaticity(mol, mf, Smo, ring, calc=calc, mci=True, av1245=True, num_threads=1)


