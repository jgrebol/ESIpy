import sys
sys.path.insert(0, "/home/joan/PycharmProjects/ESIpy")
import numpy as np
from pyscf import gto, scf
from esipy.make_aoms import make_aoms

mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='cc-pvdz')
mf = scf.RHF(mol).run(verbose=0)

aoms = make_aoms(mol, mf, 'iao')
print(f"AOM shape: {aoms[0].shape}")
for i, a in enumerate(aoms):
    print(f"Trace {i}: {np.trace(a)}")

aoms_no = make_aoms(mol, mf, 'iao', is_fchk=False)
print("done")
