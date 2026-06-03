
import numpy as np
from pyscf import gto, scf
from esipy import ESI
from esipy.tools import find_ns_unrest

# 1. Direct PySCF UHF Calculation for O2 (triplet)
mol = gto.Mole()
mol.atom = 'O 0 0 0; O 0 0 1.208'  # O2 equilibrium distance ~1.208 A
mol.basis = 'sto-3g'
mol.spin = 2
mol.build()

mf = scf.UHF(mol)
mf.kernel()

print(f"PySCF UHF converged: {mf.converged}")
print(f"MO occupations alpha: {mf.mo_occ[0]}")
print(f"MO occupations beta: {mf.mo_occ[1]}")

# 2. Try ESIpy directly with PySCF objects
try:
    print("\n--- Initializing ESI with PySCF UHF objects ---")
    # Using Mulliken partition to avoid needing NAO/IAO dependencies if they are sensitive
    esi = ESI(mol=mol, mf=mf, partition='mulliken')
    print("ESI object created successfully")
    
    aoms = esi.aom
    print(f"AOM type: {type(aoms)}")
    # For Unrest, aoms should be [list_alpha, list_beta]
    if isinstance(aoms, list) and len(aoms) == 2:
        print("AOM is a list of length 2 (Expected for Unrest)")
        print(f"Type of aoms[0]: {type(aoms[0])}")
    
    # This is where it failed for FCHK
    print("\n--- Attempting find_ns_unrest (Population) ---")
    pops = find_ns_unrest(list(range(1, mol.natm+1)), aoms[0], aoms[1])
    print(f"Pops: {pops}")
    
    # Check if we can sum them (this is what comprehensive_test.py does)
    total_pop = sum(pops)
    print(f"Total Population: {total_pop}")

except Exception as e:
    print(f"\nCaught Exception: {e}")
    import traceback
    traceback.print_exc()
