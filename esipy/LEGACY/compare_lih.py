import sys
import os
import numpy as np

# Add the parent directory to sys.path to allow imports from esipy
sys.path.append(os.path.dirname(os.getcwd()))

from esipy.readfchk import readfchk
from esipy.make_aoms import make_aoms
from esipy.iao import iao, get_num_minbas_ao
from esipy.tools import build_eta

path1 = '../joan/LiH/GS/lih_1.6.fchk'
path2 = '../joan/LiH/lih.fchk'

def run_analysis(path, label):
    print(f"\n{'='*20} Analysis for: {label} ({path}) {'='*20}")
    mol, mf, myhf = readfchk(path)
    S = mol.intor('int1e_ovlp')

    # Get minimal basis size
    n_min = sum(get_num_minbas_ao(mol, i) for i in range(mol.natm))
    print(f"Minimal basis size (n_min): {n_min}")

    occ_all = mf.mo_occ
    print(f"Total Natural Orbitals obtained: {len(occ_all)}")
    print(f"Natural Orbitals used (n_min): {n_min}")
    print(f"Top {n_min} occupations: {occ_all[:n_min]}")
    print(f"Sum of used occupations: {sum(occ_all[:n_min]):.8f}")
    print(f"Sum of ALL occupations: {sum(occ_all):.8f}")

    # 1. Pure HF
    print("\n--- Case 1: Pure HF (HF orbitals) ---")
    aom_hf = make_aoms(mol, myhf, 'iao')
    pop_hf = [2.0 * np.trace(m) for m in aom_hf]
    print(f"Li: {pop_hf[0]:.4f}, H: {pop_hf[1]:.4f}, Total: {sum(pop_hf):.4f}")

    # 2. CASSCF MOs with HF Transformation
    print("\n--- Case 2: CASSCF MOs with IAO Transformation from HF ---")
    u_nonorth_hf, pmol = iao(mol, myhf.mo_coeff[:, :n_min])
    U_hf = S @ u_nonorth_hf
    eta = build_eta(pmol)

    idx = np.argsort(mf.mo_occ)[::-1][:n_min]
    c_cas = mf.mo_coeff[:, idx]
    occ_cas = mf.mo_occ[idx]
    
    aoms_case2 = [c_cas.T @ U_hf @ eta[i] @ U_hf.T @ c_cas for i in range(mol.natm)]
    pop_case2 = [np.trace(np.diag(occ_cas) @ m) for m in aoms_case2]
    print(f"Li: {pop_case2[0]:.4f}, H: {pop_case2[1]:.4f}, Total: {sum(pop_case2):.4f}")

    # 3. Natural Orbitals (Consistent transformation)
    print("\n--- Case 3: Natural Orbitals (Consistent transf) ---")
    aom_no = make_aoms(mol, mf, 'iao')
    aoms_no, occ_no = aom_no
    pop_no = [np.trace(occ_no @ m) for m in aoms_no]
    print(f"Li: {pop_no[0]:.4f}, H: {pop_no[1]:.4f}, Total: {sum(pop_no):.4f}")
    
    return pop_no

res1 = run_analysis(path1, "GS 1.6")
res2 = run_analysis(path2, "LiH 1.6")

print("\n\n" + "#"*40)
print(f"CONSISTENCY CHECK (Case 3):")
print(f"GS 1.6:  Li={res1[0]:.6f}, H={res1[1]:.6f}")
print(f"LiH 1.6: Li={res2[0]:.6f}, H={res2[1]:.6f}")
diff = np.abs(np.array(res1) - np.array(res2)).max()
print(f"Max Difference: {diff:.2e}")
print("#"*40)
