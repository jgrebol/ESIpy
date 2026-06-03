import numpy as np
from esipy.readfchk import readfchk, read_list_from_fchk
import os

def analyze(prog, filename):
    path = f"../tests/FCHK/{prog}/{filename}"
    if not os.path.exists(path): return
    print(f"\n--- Finding Scaling: {prog} {filename} ---")
    mol_f, mf_f = readfchk(path)
    S_f = mf_f.get_ovlp()
    dt_flat = read_list_from_fchk('Total SCF Density', path)
    n = mol_f.nao
    P_raw = np.zeros((n, n))
    tril_idx = np.tril_indices(n)
    P_raw[tril_idx] = dt_flat
    P_raw = P_raw + P_raw.T - np.diag(np.diag(P_raw))
    
    # We want s such that Trace( (s * P * s) @ S ) = 42.0
    # For RHF, P_ii * S_ii = contribution to pop.
    # So s_i = sqrt( 2.0 / (P_raw_ii * S_ii) ) for an occupied orbital? No.
    # P * S should have diagonal elements that sum to N.
    diag_PS = np.diag(P_raw @ S_f)
    # This is not quite right because of off-diagonals.
    
    # Let's check the scaling of individual basis functions in S_f
    # PySCF expects S_ii = 1.0. 
    print(f"  S_f diagonal (first 10): {np.diag(S_f)[:10]}")
    # If S_ii is not 1.0, then P must be scaled by 1/sqrt(S_ii) ?
    scales = 1.0 / np.sqrt(np.diag(S_f))
    P_scaled = P_raw * scales[:, None] * scales[None, :]
    print(f"  Trace(P_scaled @ S_unit): {np.trace(P_scaled):.4f}") # Trace(P*S) where S is unit-diag
    
    # Wait, Trace(P_raw @ S_f) is the true population in Gaussian space.
    # If it's 41.07 instead of 42.0, Gaussian is using a different P.
    # Actually, for B3LYP, Nelec is ALWAYS 42.0.
    # Maybe the "Total SCF Density" is just Alpha + Beta? 
    # Benzene: 21 alpha, 21 beta.
    pass

analyze('GAUSSIAN', '1_benzene_spherical.fchk')
