import numpy as np
from pyscf import gto, scf, mp, lo

# 1. Setup and Run MP2
mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pVDZ')
mf = scf.RHF(mol).run()
mymp = mp.MP2(mf).run()

# 2. Get the correlated Density Matrix (1-RDM)
# This includes the HF density + the MP2 correlation correction
dm_mp2 = mymp.make_rdm1()

# 3. Create the NAO transformation matrix
# PySCF's lo.nao() takes (mol, mf_object, s_matrix)
# To use the correlated density, we must "inject" it into the mf object
# so the NAO module uses the correlated occupations for localization.
mf.make_rdm1 = lambda *args, **kwargs: dm_mp2
s = mol.intor_symmetric('int1e_ovlp')

# U_inv is the matrix that transforms AO basis to NAO basis
u_inv = lo.nao.nao(mol, mf, s)

# 4. Calculate Populations
# Population = Tr(D_orth) where D_orth = U' @ S @ D @ S @ U
# (Note: PySCF's lo.orth_ao usually returns U_inv, so U = inv(U_inv))
d_orth = u_inv.T @ s @ dm_mp2 @ s @ u_inv
pops = np.diag(d_orth)

print("\n--- Minimal PySCF Correlated NAO ---")
print(f"Total MP2 Electrons: {np.sum(pops):.6f}")
print(f"Oxygen Population:   {np.sum(pops[mol.search_ao_label('O')]):.4f}")
print(f"Hydrogen 1 Pop:      {np.sum(pops[mol.search_ao_label('H')[0:10]]):.4f}") # (Slice depends on basis size)
