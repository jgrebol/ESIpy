import numpy as np
from esipy.readfchk import readfchk
import os
import sys

# Ensure we can import esipy
sys.path.append(os.getcwd())

path = '../tests/FCHK/GAUSSIAN/bz.fchk'
if not os.path.exists(path):
    # Try another path if not found
    path = 'tests/FCHK/GAUSSIAN/bz.fchk'

if not os.path.exists(path):
    print(f"Error: {path} not found")
    sys.exit(1)

mol, mf = readfchk(path)
print(f"Read Benzene FCHK. Natm: {mol.natm}, Nelec: {mf.mol.nelec}")

S = mf.get_ovlp()
# Restricted coefficients
occ_mask = mf.mo_occ > 0
c = mf.mo_coeff[:, occ_mask]

print(f"MO Coeff shape (occupied): {c.shape}")
overlap_mo = c.T @ S @ c
trace_mo = np.trace(overlap_mo)
print(f"Trace(c.T @ S @ c): {trace_mo}")

# Check if it is close to 21.0
if not np.allclose(trace_mo, 21.0, atol=0.1):
    print("WARNING: MO coefficients are NOT orthonormal with respect to the PySCF overlap matrix!")
    print(f"Diagonal of c.T @ S @ c (first 5): {np.diag(overlap_mo)[:5]}")

# Now test IAO part
from pyscf.lo import iao as pyscf_iao
from pyscf.lo import orth

try:
    C_iao_nonorth = pyscf_iao.iao(mol, c)
    print(f"IAO non-orth shape: {C_iao_nonorth.shape}")
    U_nonorth = orth.vec_lowdin(C_iao_nonorth, S)
    print(f"IAO orth shape: {U_nonorth.shape}")
    
    U = S @ U_nonorth
    pmol = pyscf_iao.reference_mol(mol)
    from esipy.tools import build_eta
    eta = build_eta(pmol)
    
    aoms = [c.T @ U @ e @ U.T @ c for e in eta]
    trace_sum = np.sum([np.trace(m) for m in aoms])
    print(f"IAO Trace Sum: {trace_sum}")

except Exception as e:
    print(f"IAO Error: {e}")
    import traceback
    traceback.print_exc()

# Also check NAO
from pyscf.lo import nao
try:
    U_inv = nao.nao(mol, mf, S)
    U = np.linalg.inv(U_inv)
    from esipy.tools import build_eta
    eta_nao = build_eta(mol) # NAO uses original mol
    aoms_nao = [c.T @ U.T @ e @ U @ c for e in eta_nao]
    trace_sum_nao = np.sum([np.trace(m) for m in aoms_nao])
    print(f"NAO Trace Sum: {trace_sum_nao}")
except Exception as e:
    print(f"NAO Error: {e}")
