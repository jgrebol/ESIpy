import os
import sys
import numpy as np
import csv
import glob

# Ensure esipy is in path
sys.path.append('/home/joan/PycharmProjects/ESIpy')
from esipy import ESI
from esipy.readfchk import readfchk
from pyscf import scf
from pyscf.lo import iao as pyscf_iao
from pyscf.lo import orth

REFERENCE_CSV = '/home/joan/IAOEDU/REFERENCE/recent_actual_values.csv'

def load_reference():
    ref_data = {}
    if not os.path.exists(REFERENCE_CSV): return ref_data
    with open(REFERENCE_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['section'].lower() == 'charge' and row['method'].upper() == 'IAO':
                # Use lowercase for both stem and item keys
                key = (row['stem'].lower(), row['item'].lower())
                ref_data[key] = float(row['method_value'])
    return ref_data

def main():
    fchk_dirs = [
        "/home/joan/IAOEDU/NEWIAOS/BZ",
        "/home/joan/IAOEDU/C4H4",
        "/home/joan/IAOEDU/NEWIAOS/FORMAMIDE"
    ]
    fchk_files = []
    for d in fchk_dirs:
        fchk_files.extend(glob.glob(f"{d}/*.fchk"))
    
    ref_data = load_reference()
    results = []
    
    from pyscf.data import elements

    for fchk in sorted(fchk_files):
        stem = os.path.splitext(os.path.basename(fchk))[0].lower()
        print(f"Processing {stem}...")
        try:
            mol, mf = readfchk(fchk)
        except: 
            print(f"  Failed to read {fchk}")
            continue
        
        if not isinstance(mf, scf.hf.RHF): 
            print(f"  Skipping non-RHF: {type(mf)}")
            continue
            
        nocc = mol.nelectron // 2
        c_occ = mf.mo_coeff[:, :nocc]
        S = mol.intor('int1e_ovlp')
        
        # 1. PySCF IAO
        c_iao_pyscf = pyscf_iao.iao(mol, c_occ, minao='minao')
        c_iao_pyscf = orth.vec_lowdin(c_iao_pyscf, S)
        
        Z = [elements.charge(mol.atom_symbol(i)) for i in range(mol.natm)]
        m = c_iao_pyscf.T @ S @ c_occ
        pop_pyscf = 2.0 * np.sum(np.abs(m)**2, axis=1) # pop = 2 * trace(alpha_AOM) for restricted
        charges_pyscf = [Z[i] - pop_pyscf[i] for i in range(mol.natm)]

        # 2. ESIpy IAO
        esi = ESI(mol=mol, mf=mf, partition='iao')
        charges_esipy = [Z[i] - 2.0 * np.trace(esi.aom[i]) for i in range(mol.natm)]
        # 3. ALLEFAO check
        print(f"  Testing ALLEFAO members...")
        allefao_parts = ['iao', 'iao-autosad', 'iao-effao-lowdin', 'iao-effao-metalowdin', 'iao-effao-gross', 'iao-effao-net', 'iao-effao-sps', 'iao-effao-spsa', 'iao-effao-symmetric']
        for p in allefao_parts:
            try:
                esi_p = ESI(mol=mol, mf=mf, partition=p, rings=[])
                print(f"    {p}: success")
            except Exception as e:
                print(f"    {p}: FAILED ({e})")

        # Compare
        for i in range(mol.natm):
            sym = mol.atom_symbol(i)
            # Reference items are uppercase C1, H7 etc.
            item_ref = f"{sym.upper()}{i+1}"
            ref_val = ref_data.get((stem.lower(), item_ref.lower()), None)
            results.append({
                'stem': stem,
                'item': item_ref,
                'pyscf': charges_pyscf[i],
                'esipy': charges_esipy[i],
                'ref': ref_val
            })

    # Print Summary Table
    print(f"\n{'Stem':<30} {'Item':<6} {'PySCF':>10} {'ESIpy':>10} {'Ref':>10} {'Diff(E-P)':>10}")
    for r in results:
        diff = r['esipy'] - r['pyscf']
        ref_val = r['ref'] if r['ref'] is not None else 0.0
        print(f"{r['stem']:<30} {r['item']:<6} {r['pyscf']:10.6f} {r['esipy']:10.6f} {ref_val:10.6f} {diff:10.2e}")

if __name__ == "__main__":
    main()
