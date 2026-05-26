import numpy as np
from esipy.readfchk import readfchk
from esipy import ESI
import os
import sys

# Ensure we can import esipy
sys.path.append(os.getcwd())

path = '../tests/FCHK/GAUSSIAN/bz.fchk'
if not os.path.exists(path):
    path = 'tests/FCHK/GAUSSIAN/bz.fchk'

mol, mf = readfchk(path)
esi = ESI(mol=mol, mf=mf, partition='iao', rings=[[1,2,3,4,5,6]])

print(f"esi.aom type: {type(esi.aom)}")
if isinstance(esi.aom, list):
    print(f"esi.aom length: {len(esi.aom)}")
    if len(esi.aom) > 0:
        print(f"esi.aom[0] type: {type(esi.aom[0])}")
        if isinstance(esi.aom[0], list):
             print(f"esi.aom[0] length: {len(esi.aom[0])}")

def calculate_trace_sum(aom):
    if isinstance(aom, list) and len(aom) > 0 and isinstance(aom[0], list):
        # Unrestricted
        return sum(np.trace(m) for sublist in aom for m in sublist)
    else:
        # Restricted
        return sum(np.trace(m) for m in aom)

correct_trace_sum = calculate_trace_sum(esi.aom)
print(f"Correct trace sum: {correct_trace_sum}")
print(f"Expected: {mf.mol.nelec}") # (21, 21) -> sum is 42? 
# Wait, for Restricted, it should be 21. For Unrestricted, alpha + beta = 21 + 21 = 42.

# Let's see what the test does:
try:
    trace_sum_bad = np.sum([np.trace(m) for m in esi.aom])
    print(f"Bad trace sum (test logic): {trace_sum_bad}")
except Exception as e:
    print(f"Bad trace sum failed: {e}")
