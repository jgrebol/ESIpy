import os

code_template = '''import os
import sys
import pickle
import numpy as np
from itertools import product
from pyscf import gto, scf, mcscf
from esipy import ESI

base_dir = "/home/joan/PycharmProjects/ESIpy/joan"

FAMILIES = {
    "Original": ["mulliken", "lowdin", "meta-lowdin", "nao", "iao", "qtaim"],
    "IAO-EFFAO": ["iao", "iao-autosad", "iao-effao-gross", "iao-effao-lowdin", "iao-effao-ml", "iao-effao-symmetric", "qtaim"],
    "FPIAO": ["iao", "fpiao0.5", "fpiao0.75", "fpiao1.0", "fpiao1.25", "fpiao1.5", "fpiao1.75", "fpiao2.0", "qtaim"],
    "DFPIAO": ["iao", "dfpiao0.3", "dfpiao0.4", "dfpiao0.5", "dfpiao0.6", "dfpiao0.7", "qtaim"]
}

def extract_qtaim_di(out_file):
    if not os.path.exists(out_file): return np.nan
    with open(out_file, 'r') as f: lines = f.readlines()
    for i, line in enumerate(lines):
        if '"FUZZY ATOM" DI MATRIX' in line:
            for j in range(i+1, min(i+10, len(lines))):
                if '2  H' in lines[j] and len(lines[j].split()) >= 3:
                    try: return float(lines[j].split()[1])
                    except: pass
                elif '2  F' in lines[j] and len(lines[j].split()) >= 3:
                    try: return float(lines[j].split()[1])
                    except: pass
    return np.nan

class MockMF:
    def __init__(self, mol, mo_coeff, mo_occ, dm_ao, e_tot):
        self.mol = mol; self.mo_coeff = mo_coeff; self.mo_occ = mo_occ
        self._dm_ao = dm_ao; self.e_tot = e_tot; self.verbose = 0
    def make_rdm1(self, *args, **kwargs): return self._dm_ao
    def get_ovlp(self, *args, **kwargs): return self.mol.intor_symmetric('int1e_ovlp')
    def get_veff(self, *args, **kwargs): return np.zeros_like(self._dm_ao)

def compute_exact_di_dm2(mycas, aoms, dm1, dm2):
    ncore = getattr(mycas, 'ncore', 0)
    ncas = mycas.ncas
    xc_dens = dm2 - np.einsum('ij,kl->ijkl', dm1, dm1)
    x_dens = -0.5 * np.einsum('il,kj->ijkl', dm1, dm1)
    Smo = [np.array(aom)[ncore:ncore+ncas, ncore:ncore+ncas] for aom in aoms]
    DI_exact = -2 * np.einsum('ijkl,ji,lk->', xc_dens, Smo[0], Smo[1])
    return DI_exact

def process_lif():
    sys_key = "LiF"
    chk_dir = os.path.join(base_dir, "HARPOON", "LiF", "chk")
    os.makedirs(chk_dir, exist_ok=True)
    distances_to_run = [np.round(r, 1) for r in np.arange(1.6, 7.1, 0.1)]
    
    data_pkl = os.path.join(chk_dir, f"{sys_key}_data.pkl")
    if os.path.exists(data_pkl):
        with open(data_pkl, 'rb') as f: distances, energies, data = pickle.load(f)
    else:
        distances, energies = [], []
        data = {fam: {part: [] for part in FAMILIES[fam]} for fam in FAMILIES}

    prev_mo = None
    for dist in distances_to_run:
        if dist in distances: continue
        print(f"[{sys_key}] R = {dist:.1f} | Running SA-CASSCF(6,6)...")
        mol = gto.M(atom=f'Li 0 0 0; F 0 0 {dist}', basis='aug-cc-pVDZ', spin=0)
        mf = scf.RHF(mol)
        if prev_mo is not None:
            mf.kernel(prev_mo)
        else:
            prev_mo = np.load('lif_1.6_mo.npy')
            mf.kernel(prev_mo)
            
        mycas = mcscf.CASSCF(mf, 6, 6)
        mycas.fcisolver.nroots = 4
        mycas = mycas.state_average([0.25, 0.25, 0.25, 0.25])
        if prev_mo is not None: _, _, fcivec, mo, _ = mycas.kernel(prev_mo)
        else: _, _, fcivec, mo, _ = mycas.kernel()
        prev_mo = mo
        
        distances.append(dist)
        energies.append(mycas.e_states[0])
        
        ncore = mycas.ncore
        from pyscf import fci
        dm1, dm2 = fci.direct_spin1.make_rdm12(fcivec[0], mycas.ncas, mycas.nelecas)
        dm_ao_act = fci.direct_spin1.make_rdm1(fcivec[0], mycas.ncas, mycas.nelecas)
        dm_ao_act = mycas.mo_coeff[:, ncore:ncore+mycas.ncas] @ dm_ao_act @ mycas.mo_coeff[:, ncore:ncore+mycas.ncas].T
        mo_core = mycas.mo_coeff[:, :ncore]
        dm_ao_core = 2.0 * mo_core @ mo_core.T
        dm_ao = dm_ao_act + dm_ao_core
        
        mock_mf = MockMF(mol, mycas.mo_coeff, np.ones(mol.nao), dm_ao, mycas.e_states[0])
        mock_mf.ncore = ncore
        mock_mf.ncas = mycas.ncas
        
        # Read QTAIM
        qtaim_file = os.path.join(base_dir, "HARPOON", "LiF", f"lif_{dist:.1f}_FUZ.out")
        qtaim_di = extract_qtaim_di(qtaim_file)
        
        for fam_name, parts in FAMILIES.items():
            for part in parts:
                if part == "qtaim":
                    data[fam_name][part].append(qtaim_di)
                else:
                    try:
                        esi = ESI(mol=mol, mf=mock_mf, partition=part)
                        di_ex = compute_exact_di_dm2(mycas, esi.aom, dm1, dm2)
                        data[fam_name][part].append(di_ex)
                    except Exception as e:
                        print(f"Error {part}: {e}")
                        data[fam_name][part].append(np.nan)
        
        with open(data_pkl, 'wb') as f: pickle.dump((distances, energies, data), f)

def process_lih():
    sys_key = "LiH"
    distances_to_run = [np.round(r, 1) for r in np.arange(1.6, 7.1, 0.1)]
    
    for state, state_name in [(0, "GS"), (1, "ES")]:
        chk_dir = os.path.join(base_dir, "LiH", state_name, "chk")
        os.makedirs(chk_dir, exist_ok=True)
        data_pkl = os.path.join(chk_dir, f"{sys_key}-{state_name}_data.pkl")
        
        if os.path.exists(data_pkl):
            with open(data_pkl, 'rb') as f: d, e, data = pickle.load(f)
        else:
            d, e = [], []
            data = {fam: {part: [] for part in FAMILIES[fam]} for fam in FAMILIES}
            
        for dist in distances_to_run:
            if dist in d: continue
            print(f"[{sys_key}-{state_name}] R = {dist:.1f} | Running CASCI(4,32) / FCI...")
            mol = gto.M(atom=f'Li 0 0 0; H 0 0 {dist}', basis='aug-cc-pVDZ', spin=0, symmetry=True)
            mf = scf.RHF(mol).run()
            mycas = mcscf.CASCI(mf, 4, 32)
            mycas.fcisolver.nroots = 4
            _, _, fcivec, _, _ = mycas.kernel()
            
            d.append(dist)
            e.append(mycas.e_tot[state])
            
            from pyscf import fci
            nmo = mycas.nmo
            dm1, dm2 = fci.direct_spin1.make_rdm12(fcivec[state], nmo, mol.nelectron)
            dm_ao = mycas.mo_coeff @ fci.direct_spin1.make_rdm1(fcivec[state], nmo, mol.nelectron) @ mycas.mo_coeff.T
            
            mock_mf = MockMF(mol, mycas.mo_coeff, np.ones(nmo), dm_ao, mycas.e_tot[state])
            mock_mf.ncore = 0
            mock_mf.ncas = nmo
            
            qtaim_dir = "GS" if state == 0 else "ES"
            qtaim_file = os.path.join(base_dir, "LiH", qtaim_dir, "QTAIM", f"lih_{dist:.1f}_FUZ.out")
            qtaim_di = extract_qtaim_di(qtaim_file)
            
            for fam_name, parts in FAMILIES.items():
                for part in parts:
                    if part == "qtaim":
                        data[fam_name][part].append(qtaim_di)
                    else:
                        try:
                            esi = ESI(mol=mol, mf=mock_mf, partition=part)
                            di_ex = compute_exact_di_dm2(mycas, esi.aom, dm1, dm2)
                            data[fam_name][part].append(di_ex)
                        except Exception as ex:
                            data[fam_name][part].append(np.nan)
                            
            with open(data_pkl, 'wb') as f: pickle.dump((d, e, data), f)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "lif": process_lif()
    elif len(sys.argv) > 1 and sys.argv[1] == "lih": process_lih()
'''
with open('/home/joan/PycharmProjects/ESIpy/joan/run_new.py', 'w') as f:
    f.write(code_template)
