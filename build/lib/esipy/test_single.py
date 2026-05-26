import os
import sys
import numpy as np
import csv
from esipy import ESI
from esipy.readfchk import readfchk

sys.path.append('/home/joan/PycharmProjects/ESIpy')

def main():
    fchk = "/home/joan/IAOEDU/NEWIAOS/BZ/bz_cc-pVTZ.fchk"
    rings = [[1,2,3,4,5,6]]
    stem = "bz_cc-pVTZ"
    
    partitions = [
        'IAO', 'IAO_631G', 'IAO_CCPVDZ', 'IAO_STO3G', 'IAO_CCPVTZ',
        'DFPIAO(0.3)', 'DFPIAO(0.4)', 'DFPIAO(0.5)', 'DFPIAO(0.6)', 'DFPIAO(0.7)',
        'FPIAO(1.0)', 'FPIAO(1.25)', 'FPIAO(1.5)', 'FPIAO(1.75)', 'FPIAO(2.0)',
        'XIAO_DFPIAO(0.3)', 'XIAO_DFPIAO(0.4)', 'XIAO_DFPIAO(0.5)', 'XIAO_DFPIAO(0.6)', 'XIAO_DFPIAO(0.7)'
    ]

    print(f"Processing {stem}...")
    mol, mf = readfchk(fchk)
    
    results = []
    for part in partitions:
        print(f"  Running {part}...")
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

        esi = ESI(mol=mol, mf=mf, rings=rings, partition=actual_part, iaoref=iaoref, iaomix=iaomix)
        results.append({
            'method': part,
            'Iring': esi.indicators[0].iring,
            'MCI': esi.indicators[0].mci
        })

    with open('bz_ccpvt_test.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'Iring', 'MCI'])
        writer.writeheader()
        writer.writerows(results)
    print("Done. Saved to bz_ccpvt_test.csv")

if __name__ == "__main__":
    main()
