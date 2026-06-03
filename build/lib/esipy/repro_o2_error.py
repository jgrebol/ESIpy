
import numpy as np
from readfchk import readfchk
from esipy import ESI
from esipy.tools import find_ns_unrest

path = "../tests/FCHK/GAUSSIAN/3_o2_triplet.fchk"
mol, mf = readfchk(path)
print(f"Wavefunction type: {mf.__class__.__name__}")
print(f"MO occupations alpha: {mf.mo_occ[0]}")
print(f"MO occupations beta: {mf.mo_occ[1]}")

try:
    esi = ESI(mol=mol, mf=mf, partition='mulliken')
    print("ESI object created successfully")
    aoms = esi.aom
    print(f"AOM type: {type(aoms)}")
    if isinstance(aoms, list) and len(aoms) == 2:
        print(f"AOM length: {len(aoms)}")
        print(f"AOM[0] type: {type(aoms[0])}")

    pops = find_ns_unrest(list(range(1, mol.natm+1)), aoms[0], aoms[1])
    print(f"Pops: {pops}")
    print(f"Total Pop: {sum(pops)}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
