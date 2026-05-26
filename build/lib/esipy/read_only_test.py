import os
import sys
import numpy as np
import traceback

sys.path.append(os.getcwd())
from esipy.readfchk import readfchk

def main():
    base_dir = '../tests/FCHK'
    folders = ['GAUSSIAN', 'QCHEM']
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path): continue
            
        print(f"\n--- Testing {folder} ---")
        files = [f for f in os.listdir(folder_path) if f.endswith('.fchk')]
        files.sort()
        
        for f in files:
            path = os.path.join(folder_path, f)
            try:
                mol, mf = readfchk(path)
                print(f"[READ OK] {f:30} Nelec: {mol.nelectron}")
            except Exception as e:
                print(f"[READ FAIL] {f:30} Error: {e}")

if __name__ == '__main__':
    main()
