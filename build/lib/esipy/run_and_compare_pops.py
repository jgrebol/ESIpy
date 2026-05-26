import os
import sys
import numpy as np
import csv
import glob
import matplotlib.pyplot as plt
from esipy import ESI
from esipy.readfchk import readfchk

# Ensure esipy is in path
sys.path.append('/home/joan/PycharmProjects/ESIpy')

REFERENCE_CSV = '/home/joan/IAOEDU/REFERENCE/recent_actual_values.csv'

def run_calculation(fchk_path, partitions, rings=None):
    try:
        mol, mf = readfchk(fchk_path)
    except Exception as e:
        print(f"Error reading {fchk_path}: {e}")
        return []
        
    results = []
    # Symbols for charge calculation
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    from pyscf.data import elements
    Z = [elements.charge(s) for s in symbols]

    for part in partitions:
        print(f"  Running partition: {part}")
        iaoref = 'minao'
        iaomix = 0.5
        
        actual_part = part
        if part == 'IAO_631G': actual_part = 'iao_6-31G'
        elif part == 'IAO_CCPVDZ': actual_part = 'iao_cc-pVDZ'
        elif part == 'IAO_STO3G': actual_part = 'iao_STO-3G'
        elif part == 'IAO_CCPVTZ': actual_part = 'iao_cc-pVTZ'
        elif part.startswith('FPIAO('):
            val = float(part.split('(')[1].split(')')[0])
            actual_part = f'fpiao({val})'; iaomix = val
        elif part.startswith('DFPIAO('):
            val = float(part.split('(')[1].split(')')[0])
            actual_part = f'dfpiao({val})'; iaomix = val
        elif part.startswith('XIAO_DFPIAO('):
            val = float(part.split('(')[1].split(')')[0])
            actual_part = f'xiao_dfpiao({val})'; iaomix = val

        try:
            esi = ESI(mol=mol, mf=mf, rings=rings, partition=actual_part, iaoref=iaoref, iaomix=iaomix)
            # Extract Charges
            aoms = esi.aom
            for i in range(mol.natm):
                pop = 2 * 2 * np.trace(aoms[i]) # Multiply by 2 for spin-restricted? 
                charge = Z[i] - pop
                results.append({
                    'stem': os.path.splitext(os.path.basename(fchk_path))[0],
                    'method': part,
                    'section': 'charge',
                    'item': f"{symbols[i]}{i+1}",
                    'value': charge,
                    'pop': pop
                })
        except Exception as e:
            print(f"Error running {part} on {fchk_path}: {e}")
            
    return results

def load_reference():
    ref_data = {}
    if not os.path.exists(REFERENCE_CSV):
        print(f"Warning: Reference file {REFERENCE_CSV} not found.")
        return ref_data
    with open(REFERENCE_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['section'].lower() == 'charge':
                method = row['method'].upper()
                # Map reference method names to our benchmark method names
                if 'FPIAO(' in method and 'XIAO' not in method:
                    try:
                        v = float(method.split('(')[1].split(')')[0])
                        method = f'FPIAO({v:.1f})'
                    except: pass
                if 'XIAO_FPIAO1(' in method:
                    try:
                        v = float(method.split('(')[1].split(')')[0])
                        method = f'XIAO_DFPIAO({v:.1f})'
                    except: pass
                
                key = (row['stem'].lower(), method, row['item'].lower())
                ref_data[key] = float(row['method_value'])
    return ref_data

def main():
    # Target systems from ROBUST and ANNULENES
    # ROBUST: bz, c4h4, formamide
    # ANNULENES: benzene, c8h8, c10h10_heart, c10h10_twist, c12h12, c14h14_s, c16h16_c1, c16h16_s4, c18h18_c1
    
    fchk_files = []
    # ROBUST basis sweeps
    for d in ["/home/joan/IAOEDU/NEWIAOS/BZ", "/home/joan/IAOEDU/C4H4", "/home/joan/IAOEDU/NEWIAOS/FORMAMIDE"]:
        fchk_files.extend(glob.glob(f"{d}/*.fchk"))
    
    # ANNULENES
    fchk_files.extend(glob.glob("/home/joan/IAOEDU/ANNULENES/*.fchk"))

    # Deduplicate and sort
    fchk_files = sorted(list(set(fchk_files)))

    partitions = [
        'IAO', 'IAO_631G', 'IAO_CCPVDZ', 'IAO_STO3G', 'IAO_CCPVTZ',
        'DFPIAO(0.3)', 'DFPIAO(0.4)', 'DFPIAO(0.5)', 'DFPIAO(0.6)', 'DFPIAO(0.7)',
        'FPIAO(1.0)', 'FPIAO(1.25)', 'FPIAO(1.5)', 'FPIAO(1.75)', 'FPIAO(2.0)',
        'XIAO_DFPIAO(0.3)', 'XIAO_DFPIAO(0.4)', 'XIAO_DFPIAO(0.5)', 'XIAO_DFPIAO(0.6)', 'XIAO_DFPIAO(0.7)'
    ]

    ref_data = load_reference()
    
    results = []
    processed_stems = set()

    for fchk in fchk_files:
        stem = os.path.splitext(os.path.basename(fchk))[0]
        if stem.lower() in processed_stems: continue
        processed_stems.add(stem.lower())
        
        print(f"Processing {stem}...")
        calc_results = run_calculation(fchk, partitions)
        for res in calc_results:
            # Map method for ref lookup
            m = res['method'].upper()
            if m.startswith('FPIAO('):
                try: v = float(m.split('(')[1].split(')')[0]); m = f'FPIAO({v:.1f})'
                except: pass
            elif m.startswith('DFPIAO('):
                try: v = float(m.split('(')[1].split(')')[0]); m = f'DFPIAO({v:.1f})'
                except: pass
            elif m.startswith('XIAO_DFPIAO('):
                try: v = float(m.split('(')[1].split(')')[0]); m = f'XIAO_DFPIAO({v:.1f})'
                except: pass

            res['ref_charge'] = ref_data.get((stem.lower(), m, res['item'].lower()), None)
            results.append(res)

    # Save to CSV
    fieldnames = ['stem', 'method', 'section', 'item', 'value', 'pop', 'ref_charge']
    output_csv = 'pop_comparison_results.csv'
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(res)
            
    print(f"Population comparison results saved to {output_csv}")

    # Basic statistics
    diffs = [abs(r['value'] - r['ref_charge']) for r in results if r['ref_charge'] is not None]
    if diffs:
        print(f"Charge Max Diff: {max(diffs):.2e}")
        print(f"Charge Mean Diff: {np.mean(diffs):.2e}")
    else:
        print("No matching reference values found for charges.")

if __name__ == "__main__":
    main()
