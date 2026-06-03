from pyscf import gto, scf, mcscf, fci
mol = gto.M(atom='Li 0 0 0; F 0 0 1.6', basis='sto-3g', verbose=0)
mf = scf.RHF(mol).run()
mycas = mcscf.CASSCF(mf, 6, 6).state_average([0.25]*4)
mycas.kernel()

ci_target = mycas.ci[0]
from pyscf.fci import direct_spin1
dm1, dm2 = direct_spin1.make_rdm12(ci_target, mycas.ncas, mycas.nelecas)
print("DM1 shape:", dm1.shape)
