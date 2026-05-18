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

from esipy.iao import peiao, effao, fpiao
from esipy.make_aoms import make_aoms

def calculate_di_exact(dm1, dm2, aom_a, aom_b):
    na = np.trace(dm1 @ aom_a)
    nb = np.trace(dm1 @ aom_b)
    # dm2 indices PySCF: <p+ r+ s q> -> (p,q,r,s) where Gamma_pqrs = <p+ r+ s q>
    # DI = 2 * (Na*Nb - sum_pqrs Gamma_pqrs S^A_qp S^B_sr)
    sum_gamma = np.einsum('pqrs,qp,sr->', dm2, aom_a, aom_b, optimize=True)
    return 2.0 * (na * nb - sum_gamma)

def run_lih_comparison():
    distances = np.linspace(1.0, 5.0, 9)
    basis = 'aug-cc-pvdz'
    chk_dir = '/home/joan/PycharmProjects/ESIpy/joan/LiH/GS/PHOENIX/TEST/chk'
    
    # QTAIM reference values from previous reports in the directory
    qtaim_ref = [0.2641, 0.2009, 0.1974, 0.2567, 0.4166, 0.4536, 0.2453, 0.1035, 0.0429]
    
    # Methods to test (Natural Orbitals Approach)
    # We will use 'nao' as the reference basis for EffAOs
    methods = [
        {'name': 'IAO-EFFAO (NAO)', 'label': 'iao nao'},
        {'name': 'PEIAO (NAO)', 'label': 'peiao nao'},
        {'name': 'FPIAO(1.0) (ANO)', 'label': 'fpiao(1.0) nao'},
        {'name': 'FPIAO(1.5) (ANO)', 'label': 'fpiao(1.5) nao'},
        {'name': 'DPEIAO(0.7) (NAO)', 'label': 'dpeiao(0.7) nao'},
        {'name': 'DFPIAO(0.7) (ANO)', 'label': 'dfpiao(0.7) nao'},
    ]
    
    results = {m['name']: [] for m in methods}
    
    print(f"{'Dist':<6} | {'QTAIM':<8} | " + " | ".join([f"{m['name'][:10]:<10}" for m in methods]))
    print("-" * 100)

    for i, dist in enumerate(distances):
        chk_file = os.path.join(chk_dir, f"lih_{basis}_{dist:.2f}_fci.chk")
        mol = gto.M(atom=f'Li 0 0 0; H 0 0 {dist}', basis=basis, verbose=0)
        
        if not os.path.exists(chk_file): continue
        mf_data = scf.chkfile.load(chk_file, 'scf')
        fci_data = scf.chkfile.load(chk_file, 'fci')
        if mf_data is None or fci_data is None: continue

        mf = scf.RHF(mol)
        mf.__dict__.update(mf_data)
        fcivec = fci_data['vector']
        cisolver = fci.FCI(mf)
        dm1_mo = cisolver.make_rdm1(fcivec, mol.nao, mol.nelec)
        dm2_mo = cisolver.make_rdm2(fcivec, mol.nao, mol.nelec)
        
        # Compute NOs and dm2 in NO basis once
        occ_no, no_mo_rot = np.linalg.eigh(dm1_mo)
        idx = occ_no.argsort()[::-1]
        occ_no = occ_no[idx]; no_mo_rot = no_mo_rot[:, idx]
        dm1_no = np.diag(occ_no)
        # Pre-transform dm2 to NO basis
        dm2_no = np.einsum('pi,qj,rk,sl,pqrs->ijkl', no_mo_rot, no_mo_rot, no_mo_rot, no_mo_rot, dm2_mo)
        
        row = [f"{dist:<6.2f}", f"{qtaim_ref[i]:<8.4f}"]
        
        class MockMF(scf.hf.SCF):
            def __init__(self, mol, mf_orig, dm1_mo):
                self.mol = mol
                self.mo_coeff = mf_orig.mo_coeff
                self.dm1_mo = dm1_mo
                self.verbose = 0
            def make_rdm1(self, ao_repr=True):
                if ao_repr: return self.mo_coeff @ self.dm1_mo @ self.mo_coeff.T
                return self.dm1_mo
            def get_ovlp(self): return self.mol.intor('int1e_ovlp')

        mock_mf = MockMF(mol, mf, dm1_mo)
        
        for m in methods:
            aoms_no = make_aoms(mol, mock_mf, m['label'])
            di = calculate_di_exact(dm1_no, dm2_no, aoms_no[0], aoms_no[1])
            results[m['name']].append(di)
            row.append(f"{di:<10.4f}")
            
        print(" | ".join(row))

    # Plotting
    plt.figure(figsize=(10, 7))
    plt.plot(distances, qtaim_ref, 'k--', label='QTAIM', linewidth=2, marker='x')
    
    markers = ['o', 's', '^', 'v', 'D', 'p']
    for idx, m in enumerate(methods):
        plt.plot(distances, results[m['name']], label=m['name'], marker=markers[idx % len(markers)])
        
    plt.title("LiH Dissociation: PEIAO/FPIAO vs QTAIM (FCI/NO Approach)")
    plt.xlabel("Distance (Å)")
    plt.ylabel("Delocalization Index (DI)")
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig("lih_comparison_peiao_fpiiao.png")
    print("\nSaved plot: lih_comparison_peiao_fpiiao.png")

if __name__ == "__main__":
    run_lih_comparison()
