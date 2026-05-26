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
    for part in partitions:
        print(f"  Running partition: {part}")
        iaoref = 'minao'
        iaomix = 0.5
        
        actual_part = part
        if part == 'IAO_631G':
            actual_part = 'iao_6-31G'
        elif part == 'IAO_CCPVDZ':
            actual_part = 'iao_cc-pVDZ'
        elif part == 'IAO_STO3G':
            actual_part = 'iao_STO-3G'
        elif part == 'IAO_CCPVTZ':
            actual_part = 'iao_cc-pVTZ'
        elif part.startswith('FPIAO('):
            val = float(part.split('(')[1].split(')')[0])
            actual_part = f'fpiao({val})'
            iaomix = val
        elif part.startswith('DFPIAO('):
            val = float(part.split('(')[1].split(')')[0])
            actual_part = f'dfpiao({val})'
            iaomix = val
        elif part.startswith('XIAO_DFPIAO('):
            val = float(part.split('(')[1].split(')')[0])
            # The user said 0.7 in output may be our 0.3 due to x <-> 1-x mix
            actual_part = f'xiao_dfpiao({val})'
            iaomix = val

        try:
            # For precision, use the object directly
            # For annulenes, we might need specific findrings logic if not provided
            esi = ESI(mol=mol, mf=mf, rings=rings, partition=actual_part, iaoref=iaoref, iaomix=iaomix)
            
            if rings:
                for idx, ind in enumerate(esi.indicators):
                    results.append({
                        'stem': os.path.splitext(os.path.basename(fchk_path))[0],
                        'method': part,
                        'Iring': ind.iring,
                        'MCI': ind.mci
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
            if row['quantity'] in ['Iring', 'MCI']:
                method = row['method']
                # Normalizing method names for comparison
                if method == 'IAO_631G': method = 'IAO_631G'
                elif method == 'IAO_CCPVDZ': method = 'IAO_CCPVDZ'
                elif method == 'IAO_STO3G': method = 'IAO_STO3G'
                elif method == 'IAO_CCPVTZ': method = 'IAO_CCPVTZ'
                elif method.startswith('FPIAO('):
                    try:
                        v = float(method.split('(')[1].split(')')[0])
                        method = f'FPIAO({v:.1f})'
                    except: pass
                elif method.startswith('XIAO_FPIAO1('):
                    try:
                        v = float(method.split('(')[1].split(')')[0])
                        # Map to our XIAO_DFPIAO for comparison
                        method = f'XIAO_DFPIAO({v:.1f})'
                    except: pass
                
                key = (row['stem'], method, row['quantity'])
                ref_data[key] = float(row['method_value'])
    return ref_data

def main():
    # ANNULENE Rings
    annulene_rings = {
        'benzene': [[1,2,3,4,5,6]],
        'bz': [[1,2,3,4,5,6]],
        'c4h4': [[1,2,3,4]],
        'c8h8': [[1,2,3,4,5,6,7,8]],
        'c10h10_heart': [[1,2,3,4,5,6,7,8,9,10]],
        'c10h10_twist': [[1,2,3,4,5,6,7,8,9,10]],
        'c12h12': [[1,2,3,4,5,6,7,8,9,10,11,12]],
        'c14h14_s': [[1,2,3,4,5,6,7,8,9,10,11,12,13,14]],
        'c16h16_c1': [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]],
        'c16h16_s4': [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]],
        'c18h18_c1': [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
    }

    fchk_files = []
    # 1. Annulenes
    for f in glob.glob("/home/joan/IAOEDU/ANNULENES/*.fchk"):
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem in annulene_rings:
            fchk_files.append((f, annulene_rings[stem]))
    
    # 2. ROBUST sets (BZ, C4H4)
    # The .hs files in ROBUST point to NEWIAOS/BZ and C4H4
    for f in glob.glob("/home/joan/IAOEDU/NEWIAOS/BZ/*.fchk"):
        stem = os.path.splitext(os.path.basename(f))[0]
        fchk_files.append((f, [[1,2,3,4,5,6]]))
    for f in glob.glob("/home/joan/IAOEDU/C4H4/*.fchk"):
        # Filter only the basic c4h4 or basis sweeps
        stem = os.path.splitext(os.path.basename(f))[0]
        if 'c4h4' in stem:
            fchk_files.append((f, [[1,2,3,4]]))

    partitions = [
        'IAO', 'IAO_631G', 'IAO_CCPVDZ', 'IAO_STO3G', 'IAO_CCPVTZ',
        'DFPIAO(0.3)', 'DFPIAO(0.4)', 'DFPIAO(0.5)', 'DFPIAO(0.6)', 'DFPIAO(0.7)',
        'FPIAO(1.0)', 'FPIAO(1.25)', 'FPIAO(1.5)', 'FPIAO(1.75)', 'FPIAO(2.0)',
        'XIAO_DFPIAO(0.3)', 'XIAO_DFPIAO(0.4)', 'XIAO_DFPIAO(0.5)', 'XIAO_DFPIAO(0.6)', 'XIAO_DFPIAO(0.7)'
    ]

    ref_data = load_reference()
    
    results = []
    processed_stems = set()

    for fchk, rings in fchk_files:
        stem = os.path.splitext(os.path.basename(fchk))[0]
        if stem in processed_stems: continue
        processed_stems.add(stem)
        
        print(f"Processing {stem}...")
        calc_results = run_calculation(fchk, partitions, rings)
        for res in calc_results:
            # Normalizing method for ref lookup
            m = res['method']
            if m.startswith('FPIAO('):
                try: v = float(m.split('(')[1].split(')')[0]); m = f'FPIAO({v:.1f})'
                except: pass
            elif m.startswith('DFPIAO('):
                try: v = float(m.split('(')[1].split(')')[0]); m = f'DFPIAO({v:.1f})'
                except: pass
            elif m.startswith('XIAO_DFPIAO('):
                try: v = float(m.split('(')[1].split(')')[0]); m = f'XIAO_DFPIAO({v:.1f})'
                except: pass

            res['Iring_ref'] = ref_data.get((stem, m, 'Iring'), None)
            res['MCI_ref'] = ref_data.get((stem, m, 'MCI'), None)
            results.append(res)

    # Save to CSV
    fieldnames = ['stem', 'method', 'Iring', 'Iring_ref', 'MCI', 'MCI_ref']
    output_csv = 'comparison_results_final.csv'
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(res)
            
    print(f"Comparison results saved to {output_csv}")

    # Report and Graphs
    for plot_stem in ['bz_cc-pVTZ', 'c4h4_cc-pVTZ', 'benzene']:
        plot_data = [r for r in results if r['stem'] == plot_stem and r['Iring_ref'] is not None]
        if not plot_data: continue
        
        methods = [r['method'] for r in plot_data]
        iring_calc = [r['Iring'] for r in plot_data]
        iring_ref = [r['Iring_ref'] for r in plot_data]
        
        plt.figure(figsize=(12,7))
        x_indices = range(len(methods))
        plt.plot(x_indices, iring_calc, 'bo-', label='Calculated')
        plt.plot(x_indices, iring_ref, 'rx--', label='Reference')
        plt.xticks(x_indices, methods, rotation=45, ha='right')
        plt.title(f"Iring comparison for {plot_stem}")
        plt.ylabel("Iring")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{plot_stem}_iring_comparison.png")
        print(f"Generated plot: {plot_stem}_iring_comparison.png")

if __name__ == "__main__":
    main()
