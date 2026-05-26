import numpy as np
import scipy.linalg
import os
import sys
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci
import warnings

# Suppress PySCF warnings
warnings.filterwarnings("ignore")
plt.switch_backend('Agg')

# Setup paths to import esipy
project_root = "/home/joan/PycharmProjects/ESIpy"
if project_root not in sys.path:
    sys.path.append(project_root)

from esipy.iao import peiao, effao, fpiao, iao, fpiao_effao

def transform_dm2(dm2, U):
    res = np.einsum('pi,pqrs->iqrs', U, dm2, optimize=True)
    res = np.einsum('qj,iqrs->ijrs', U, res, optimize=True)
    res = np.einsum('rk,ijrs->ijks', U, res, optimize=True)
    res = np.einsum('sl,ijks->ijkl', U, res, optimize=True)
    return res

def calculate_di(dm1_iao, dm2_iao, slc_a, slc_b):
    na = np.trace(dm1_iao[np.ix_(slc_a, slc_a)])
    nb = np.trace(dm1_iao[np.ix_(slc_b, slc_b)])
    sum_gamma = np.einsum('iikk->', dm2_iao[np.ix_(slc_a, slc_a, slc_b, slc_b)], optimize=True)
    return 2.0 * (na * nb - sum_gamma)

def run_lih_comparison():
    distances = np.linspace(1.0, 5.0, 9)
    basis = 'aug-cc-pvdz'
    chk_dir = '/home/joan/PycharmProjects/ESIpy/joan/LiH/GS/PHOENIX/TEST/chk'
    
    # QTAIM reference values
    qtaim_ref = [0.2641, 0.2009, 0.1974, 0.2567, 0.4166, 0.4536, 0.2453, 0.1035, 0.0429]
    
    # Methods to test (Refined for consistency with T09)
    # Using heavy_only=False to match T09 values (0.49 for FPIAO 1.0)
    methods = [
        {'name': 'IAO-EFFAO (NAO)', 'label': 'iao nao', 'type': 'iao-effao', 'heavy': False},
        {'name': 'PEIAO NAO', 'label': 'peiao nao', 'type': 'peiao', 'heavy': False},
        {'name': 'FPIAO(1.0) NAO', 'label': 'fpiao(1.0) nao', 'type': 'fpiao', 'heavy': False, 'x': 1.0},
        {'name': 'FPIAO(1.5) NAO', 'label': 'fpiao(1.5) nao', 'type': 'fpiao', 'heavy': False, 'x': 1.5},
        {'name': 'FPIAO(1.75) NAO', 'label': 'fpiao(1.75) nao', 'type': 'fpiao', 'heavy': False, 'x': 1.75},
        {'name': 'FPIAO(2.0) NAO', 'label': 'fpiao(2.0) nao', 'type': 'fpiao', 'heavy': False, 'x': 2.0},
        {'name': 'DPEIAO(0.7) NAO', 'label': 'dpeiao(0.7) nao', 'type': 'dpeiao', 'heavy': False, 'weight': 0.7},
        {'name': 'DFPIAO(0.7) NAO', 'label': 'dfpiao(0.7) nao', 'type': 'dfpiao', 'heavy': False, 'weight': 0.7, 'x': 1.0},
    ]
    
    results = {m['name']: [] for m in methods}
    
    print(f"{'Dist':<6} | {'QTAIM':<8} | " + " | ".join([f"{m['name'][:10]:<10}" for m in methods]))
    print("-" * 140)

    for i_dist, dist in enumerate(distances):
        chk_file = os.path.join(chk_dir, f"lih_{basis}_{dist:.2f}_fci.chk")
        if not os.path.exists(chk_file): continue
        
        mol = gto.M(atom=f'Li 0 0 0; H 0 0 {dist}', basis=basis, verbose=0)
        mf_data = scf.chkfile.load(chk_file, 'scf')
        fci_data = scf.chkfile.load(chk_file, 'fci')
        if mf_data is None or fci_data is None: continue

        mf = scf.RHF(mol)
        mf.__dict__.update(mf_data)
        fcivec = fci_data['vector']
        cisolver = fci.FCI(mf)
        
        dm1_mo = cisolver.make_rdm1(fcivec, mol.nao, mol.nelec)
        dm2_mo = cisolver.make_rdm2(fcivec, mol.nao, mol.nelec)
        S = mol.intor('int1e_ovlp')
        C_mo = mf.mo_coeff

        occ_no, no_mo_rot = np.linalg.eigh(dm1_mo)
        idx_no = occ_no.argsort()[::-1]
        C_no = C_mo @ no_mo_rot[:, idx_no]
        ref_coeffs = C_no[:, :6]
        
        row_vals = [f"{dist:<6.2f}", f"{qtaim_ref[i_dist]:<8.4f}"]

        for m in methods:
            def get_di_for_iao(c_iao, pmol):
                U = C_mo.T @ S @ c_iao
                dm1_iao = U.T @ dm1_mo @ U
                dm2_iao = transform_dm2(dm2_mo, U)
                slc_a = np.arange(pmol.aoslice_by_atom()[0, 2], pmol.aoslice_by_atom()[0, 3])
                slc_b = np.arange(pmol.aoslice_by_atom()[1, 2], pmol.aoslice_by_atom()[1, 3])
                return calculate_di(dm1_iao, dm2_iao, slc_a, slc_b)

            if m['type'] == 'iao-effao':
                c_iao, pmol = effao(mol, ref_coeffs, mode='nao', polarized=False, mf=mf, heavy_only=m['heavy'])
                di = get_di_for_iao(c_iao, pmol)
            elif m['type'] == 'peiao':
                c_iao, pmol = peiao(mol, ref_coeffs, mode='nao', mf=mf, heavy_only=m['heavy'])
                di = get_di_for_iao(c_iao, pmol)
            elif m['type'] == 'fpiao':
                c_iao, pmol = fpiao_effao(mol, ref_coeffs, x=m['x'], mode='nao', pol_basis='ano', mf=mf, heavy_only=m['heavy'])
                di = get_di_for_iao(c_iao, pmol)
            elif m['type'] == 'dpeiao':
                c_min, pmol_min = effao(mol, ref_coeffs, mode='nao', polarized=False, mf=mf, heavy_only=m['heavy'])
                c_pol, pmol_pol = peiao(mol, ref_coeffs, mode='nao', mf=mf, heavy_only=m['heavy'])
                di = m['weight'] * get_di_for_iao(c_min, pmol_min) + (1.0 - m['weight']) * get_di_for_iao(c_pol, pmol_pol)
            elif m['type'] == 'dfpiao':
                c_min, pmol_min = effao(mol, ref_coeffs, mode='nao', polarized=False, mf=mf, heavy_only=m['heavy'])
                c_pol, pmol_pol = fpiao_effao(mol, ref_coeffs, x=m['x'], mode='nao', pol_basis='ano', mf=mf, heavy_only=m['heavy'])
                di = m['weight'] * get_di_for_iao(c_min, pmol_min) + (1.0 - m['weight']) * get_di_for_iao(c_pol, pmol_pol)
            
            results[m['name']].append(di)
            row_vals.append(f"{di:<10.4f}")
        
        print(" | ".join(row_vals))

    # Plotting
    plt.figure(figsize=(10, 7))
    plt.plot(distances, qtaim_ref, 'k--', label='QTAIM', linewidth=2, marker='x')
    
    markers = ['o', 's', '^', 'v', '>', '<', 'D', 'p']
    for idx, m in enumerate(methods):
        plt.plot(distances, results[m['name']], label=m['name'], marker=markers[idx % len(markers)])
        
    plt.title("LiH Dissociation: PEIAO/FPIAO/DFPIAO Benchmark (FCI, NO-Approach)")
    plt.xlabel("Distance (Å)")
    plt.ylabel("Delocalization Index (DI)")
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("lih_peiao_fpiao_benchmark.png", dpi=300)
    print("\nPlot updated: lih_peiao_fpiao_benchmark.png")

if __name__ == "__main__":
    run_lih_comparison()
