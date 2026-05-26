from pyscf import gto, dft
mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', spin=2, charge=0)
mol.build()
mf = dft.UKS(mol)
mf.xc = 'B3LYP'
mf.kernel()
print(type(mf.mo_coeff))
if isinstance(mf.mo_coeff, (list, tuple)):
    print(len(mf.mo_coeff))
if hasattr(mf.mo_coeff, 'shape'):
    print(mf.mo_coeff.shape)
