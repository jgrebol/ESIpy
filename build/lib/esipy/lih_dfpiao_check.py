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

from esipy.iao import fpiao_effao, effao

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

def verify_dfpiao():
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
    
    # 1. IAO-EFFAO-NAO
    c_iao_min, pmol_min = effao(mol, ref_coeffs, mode='nao', polarized=False, mf=mf)
    U_min = C_mo.T @ S @ c_iao_min
    dm1_iao_min = U_min.T @ dm1_mo @ U_min
    dm2_iao_min = transform_dm2(dm2_mo, U_min)
    slc_a_min = np.arange(pmol_min.aoslice_by_atom()[0, 2], pmol_min.aoslice_by_atom()[0, 3])
    slc_b_min = np.arange(pmol_min.aoslice_by_atom()[1, 2], pmol_min.aoslice_by_atom()[1, 3])
    di_min = calculate_di(dm1_iao_min, dm2_iao_min, slc_a_min, slc_b_min)
    
    # 2. FPIAO(1.0) NAO
    c_iao_pol, pmol_pol = fpiao_effao(mol, ref_coeffs, x=1.0, mode='nao', pol_basis='ano', mf=mf, heavy_only=True)
    U_pol = C_mo.T @ S @ c_iao_pol
    dm1_iao_pol = U_pol.T @ dm1_mo @ U_pol
    dm2_iao_pol = transform_dm2(dm2_mo, U_pol)
    slc_a_pol = np.arange(pmol_pol.aoslice_by_atom()[0, 2], pmol_pol.aoslice_by_atom()[0, 3])
    slc_b_pol = np.arange(pmol_pol.aoslice_by_atom()[1, 2], pmol_pol.aoslice_by_atom()[1, 3])
    di_pol = calculate_di(dm1_iao_pol, dm2_iao_pol, slc_a_pol, slc_b_pol)
    
    # 3. DFPIAO(0.7) NAO
    di_dfpiao = 0.7 * di_min + 0.3 * di_pol
    
    print(f"IAO-EFFAO DI: {di_min:.4f}")
    print(f"FPIAO(1.0) DI: {di_pol:.4f}")
    print(f"DFPIAO(0.7) DI: {di_dfpiao:.4f}")

if __name__ == "__main__":
    verify_dfpiao()
