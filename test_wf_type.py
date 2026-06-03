import sys
sys.path.insert(0, "/home/joan/PycharmProjects/ESIpy")
import numpy as np
from pyscf import gto, scf
from esipy.make_aoms import make_aoms
from esipy.tools import wf_type

mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='cc-pVDZ', spin=1, charge=1)
mf = scf.UHF(mol).run(verbose=0)
aom = make_aoms(mol, mf, 'iao-effao-sym')

print(f"type(aom): {type(aom)}")
print(f"len(aom): {len(aom)}")
print(f"type(aom[0]): {type(aom[0])}")
if isinstance(aom[0], list):
    print(f"len(aom[0]): {len(aom[0])}")
    print(f"type(aom[0][0]): {type(aom[0][0])}")
try:
    print(wf_type(aom))
except Exception as e:
    print(e)
