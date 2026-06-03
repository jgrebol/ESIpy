import sys
sys.path.insert(0, "/home/joan/PycharmProjects/ESIpy")
import numpy as np
from pyscf import gto, scf
from esipy.make_aoms import make_aoms

mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='cc-pvdz')
mf = scf.RHF(mol).run(verbose=0)
p = 0.6

aom_iao = make_aoms(mol, mf, 'iao')
aom_fpiao = make_aoms(mol, mf, 'fpiao(1.0)')
aom_dfpiao = make_aoms(mol, mf, f'dfpiao({p})')

for i in range(mol.natm):
    expected = p * aom_iao[i] + (1.0 - p) * aom_fpiao[i]
    diff = np.max(np.abs(aom_dfpiao[i] - expected))
    print(f"Atom {i} max diff: {diff}")
