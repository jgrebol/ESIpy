import numpy as np
import scipy.linalg
import os
import sys
from pyscf import gto, scf, fci
import warnings

# Suppress PySCF warnings
warnings.filterwarnings("ignore")

# Setup paths to import esipy
project_root = "/home/joan/PycharmProjects/ESIpy"
if project_root not in sys.path:
    sys.path.append(project_root)

from esipy.iao import fpiao_effao

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

def verify_consistency():
    dist = 1.0
    basis = 'aug-cc-pvdz'
    chk_file = f'/home/joan/PycharmProjects/ESIpy/joan/LiH/GS/PHOENIX/TEST/chk/lih_{basis}_{dist:.2f}_fci.chk'
    
    mol = gto.M(atom=f'Li 0 0 0; H 0 0 {dist}', basis=basis, verbose=0)
    mf_data = scf.chkfile.load(chk_file, 'scf')
    fci_data = scf.chkfile.load(chk_file, 'fci')
    
    mf = scf.RHF(mol)
    mf.__dict__.update(mf_data)
    fcivec = fci_data['vector']
    cisolver = fci.FCI(mf)
    
    dm1_mo = cisolver.make_rdm1(fcivec, mol.nao, mol.nelec)
    dm2_mo = cisolver.make_rdm2(fcivec, mol.nao, mol.nelec)
    S = mol.intor('int1e_ovlp')
    C_mo = mf.mo_coeff

    # Natural Orbitals (n=6)
    occ_no, no_mo_rot = np.linalg.eigh(dm1_mo)
    idx_no = occ_no.argsort()[::-1]
    C_no = C_mo @ no_mo_rot[:, idx_no]
    ref_coeffs = C_no[:, :6]
    
    print(f"{'x':<6} | {'DI':<10} | {'pmol.nao':<10}")
    print("-" * 30)

    for x in [1.0, 1.25, 1.5, 1.75, 2.0]:
        c_iao, pmol = fpiao_effao(mol, ref_coeffs, x=x, mode='nao', pol_basis='ano', mf=mf, heavy_only=False)
        U = C_mo.T @ S @ c_iao
        dm1_iao = U.T @ dm1_mo @ U
        dm2_iao = transform_dm2(dm2_mo, U)
        slc_a = np.arange(pmol.aoslice_by_atom()[0, 2], pmol.aoslice_by_atom()[0, 3])
        slc_b = np.arange(pmol.aoslice_by_atom()[1, 2], pmol.aoslice_by_atom()[1, 3])
        di = calculate_di(dm1_iao, dm2_iao, slc_a, slc_b)
        print(f"{x:<6.2f} | {di:<10.4f} | {pmol.nao:<10}")

    print("\nReported in T09-LiH for PEIAO-NAO (n_no=6):")
    print("x=1.5 : 0.4900")
    print("x=2.0 : 0.4717")

if __name__ == "__main__":
    verify_consistency()
