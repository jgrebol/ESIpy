from tests.test5_casscf_indicators import ESItest, mol, mf, myhf, ring
import numpy as np
from esipy import ESI
from esipy.tools import wf_type

esitest = ESI(mol=mol, mf=mf, myhf=myhf, rings=ring, partition='iao')
print("type of aom:", type(esitest.aom))
print("len of aom:", len(esitest.aom))
print("type of aom[0]:", type(esitest.aom[0]))
print("type of aom[1]:", type(esitest.aom[1]))
try:
    print("wf_type:", wf_type(esitest.aom))
except Exception as e:
    print("Error:", e)
