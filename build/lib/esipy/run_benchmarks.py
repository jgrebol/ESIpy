import os
import sys
import numpy as np
import csv
import glob
from esipy import ESI
from esipy.readfchk import readfchk

# Ensure esipy is in path
sys.path.append('/home/joan/PycharmProjects/ESIpy')

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
        # Handle special cases for reference basis
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
            # The user said 0.7 in output may be our 0.3
            # make_aoms.py: xiao_dfpiao uses (1-w)*IAO + w*FPIAO
            actual_part = f'xiao_dfpiao({val})'
            iaomix = val

        try:
            esi = ESI(mol=mol, mf=mf, rings=rings, partition=actual_part, iaoref=iaoref, iaomix=iaomix)
            
            # Extract Charges
            aoms = esi.aom
            for i in range(mol.natm):
                pop = 2 * np.trace(aoms[i])
                charge = Z[i] - pop
                results.append({
                    'stem': os.path.splitext(os.path.basename(fchk_path))[0],
                    'method': part,
                    'section': 'charge',
                    'item': f"{symbols[i]}{i+1}",
                    'value': charge
                })
            
            # Extract DIs
            for i in range(mol.natm):
                for j in range(i + 1, mol.natm):
                    di = 2 * np.einsum('ij,ji->', aoms[i], aoms[j])
                    results.append({
                        'stem': os.path.splitext(os.path.basename(fchk_path))[0],
                        'method': part,
                        'section': 'di',
                        'item': f"{symbols[i]}{i+1}-{symbols[j]}{j+1}",
                        'value': di
                    })
            
            # Extract Ring Indices
            if rings:
                for idx, ind in enumerate(esi.indicators):
                    results.append({
                        'stem': os.path.splitext(os.path.basename(fchk_path))[0],
                        'method': part,
                        'section': 'ring',
                        'item': 'Iring',
                        'value': ind.iring
                    })
                    results.append({
                        'stem': os.path.splitext(os.path.basename(fchk_path))[0],
                        'method': part,
                        'section': 'ring',
                        'item': 'MCI',
                        'value': ind.mci
                    })
        except Exception as e:
            print(f"Error running {part} on {fchk_path}: {e}")
            
    return results

def main():
    # ROBUST set
    bz_files = glob.glob("/home/joan/IAOEDU/NEWIAOS/BZ/*.fchk")
    c4h4_files = glob.glob("/home/joan/IAOEDU/C4H4/c4h4_*.fchk")
    formamide_files = glob.glob("/home/joan/IAOEDU/NEWIAOS/FORMAMIDE/*.fchk")
    
    fieldnames = ['stem', 'method', 'section', 'item', 'value']
    with open('benchmark_results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    all_fchks = []
    for f in bz_files:
        all_fchks.append((f, [[1,2,3,4,5,6]]))
    for f in c4h4_files:
        all_fchks.append((f, [[1,2,3,4]]))
    for f in formamide_files:
        all_fchks.append((f, None))
        
    partitions = [
        'IAO', 'IAO_631G', 'IAO_CCPVDZ', 'IAO_STO3G', 'IAO_CCPVTZ',
        'DFPIAO(0.3)', 'DFPIAO(0.4)', 'DFPIAO(0.5)', 'DFPIAO(0.6)', 'DFPIAO(0.7)',
        'FPIAO(1.0)', 'FPIAO(1.25)', 'FPIAO(1.5)', 'FPIAO(1.75)', 'FPIAO(2.0)',
        'XIAO_DFPIAO(0.3)', 'XIAO_DFPIAO(0.4)', 'XIAO_DFPIAO(0.5)', 'XIAO_DFPIAO(0.6)', 'XIAO_DFPIAO(0.7)'
    ]
    
    for fchk, rings in all_fchks:
        print(f"Processing {fchk}...")
        results = run_calculation(fchk, partitions, rings)
        with open('benchmark_results.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for res in results:
                writer.writerow(res)
    
    print("Saved results to benchmark_results.csv")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"CRITICAL ERROR: {e}")
        traceback.print_exc()
