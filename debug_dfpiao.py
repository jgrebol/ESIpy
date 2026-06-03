import numpy as np
from pyscf import gto, scf
from esipy.make_aoms import make_aoms

mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run()

aom_iao = make_aoms(mol, mf, 'iao-effao-nao')
aom_fpiao = make_aoms(mol, mf, 'fpiao minao(1.0)')
aom_dfpiao = make_aoms(mol, mf, 'dfpiao minao(0.6)')

expected = 0.6 * aom_iao[0] + 0.4 * aom_fpiao[0]
actual = aom_dfpiao[0]

print("Expected [0,0]:", expected[0,0])
print("Actual   [0,0]:", actual[0,0])
print("Max diff:", np.max(np.abs(expected - actual)))

# Let's also check fpiao(1.0) vs fpiao inside
# We can't easily hook inside, but we can print their trace
print("Trace iao:   ", np.trace(aom_iao[0]))
print("Trace fpiao: ", np.trace(aom_fpiao[0]))
print("Trace dfpiao:", np.trace(aom_dfpiao[0]))

