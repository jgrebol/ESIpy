import numpy as np
from pyscf import gto, scf
from esipy.make_aoms import make_aoms

mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run()

aom_iao = make_aoms(mol, mf, 'iao-effao-nao')
aom_peiao = make_aoms(mol, mf, 'peiao minao')
aom_dpeiao = make_aoms(mol, mf, 'dpeiao minao(0.2)')

expected = 0.2 * aom_iao[0] + 0.8 * aom_peiao[0]
actual = aom_dpeiao[0]

print("Expected [0,0]:", expected[0,0])
print("Actual   [0,0]:", actual[0,0])
print("Max diff:", np.max(np.abs(expected - actual)))

print("Trace iao:   ", np.trace(aom_iao[0]))
print("Trace peiao: ", np.trace(aom_peiao[0]))
print("Trace dpeiao:", np.trace(aom_dpeiao[0]))

