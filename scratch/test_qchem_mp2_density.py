from esipy import ESI
from esipy.readfchk import readfchk
import numpy as np
import pickle

q_path = '/home/joan/PycharmProjects/ESIpy/tests/FCHK/QCHEM/7_rmp2.fchk'
g_path = '/home/joan/PycharmProjects/ESIpy/tests/FCHK/GAUSSIAN/7_rmp2.fchk'

mol_q, mf_q = readfchk(q_path)
mol_g, mf_g = readfchk(g_path)

esi_q = ESI(mol=mol_q, mf=mf_q, partition='mulliken')
esi_g = ESI(mol=mol_g, mf=mf_g, partition='mulliken')

from esipy.tools import find_ns

pop_q = np.array(find_ns(list(range(1, mol_q.natm+1)), esi_q.aom))
pop_g = np.array(find_ns(list(range(1, mol_g.natm+1)), esi_g.aom))

print("Gaussian MP2 Pop: ", pop_g)
print("Q-Chem 'SCF' Pop: ", pop_q)

# Let's also compare to PySCF RHF
from pyscf import scf
mf_rhf = scf.RHF(mol_q).run()
esi_rhf = ESI(mol=mol_q, mf=mf_rhf, partition='mulliken')
pop_rhf = np.array(find_ns(list(range(1, mol_q.natm+1)), esi_rhf.aom))
print("PySCF RHF Pop:    ", pop_rhf)
print("Diff Q-Chem vs PySCF RHF:", np.max(np.abs(pop_q - pop_rhf)))

