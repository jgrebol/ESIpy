from pyscf import gto, scf
from esipy.make_aoms import make_aoms

mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='cc-pvdz')
mf = scf.RHF(mol).run(verbose=0)
aom = make_aoms(mol, mf, 'iao')
print("RHF AOM shape:", aom[0].shape)
print("Occupied MOs:", sum(mf.mo_occ > 0))

mf_u = scf.UHF(mol).run(verbose=0)
aom_u = make_aoms(mol, mf_u, 'iao')
print("UHF AOM type:", type(aom_u))
if isinstance(aom_u, list):
    print("UHF AOM list length:", len(aom_u))
    print("UHF AOM_a shape:", aom_u[0][0].shape)
elif isinstance(aom_u, tuple):
    print("UHF AOM tuple length:", len(aom_u))
    print("UHF AOM_a shape:", aom_u[0][0].shape)
