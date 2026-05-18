import os
import numpy as np
from pyscf import gto, scf

class Input:
    def __init__(self):
        self.fchk = None
        self.partition = []
        self.fragments = []
        self.save = None
        self.iaomix = 0.5
        self.iaoref = 'minao'
        self.iaopol = 'ano'
        self.heavy_only = None
        self.full_basis = False

def read_input(path):
    obj = Input()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file {path} not found")

    with open(path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue

        if line.startswith('$'):
            if line.startswith('$FCHK'):
                i += 1
                if i < len(lines):
                    obj.fchk = lines[i].strip()
            elif line.startswith('$SAVE'):
                i += 1
                if i < len(lines):
                    obj.save = lines[i].strip()
            elif line.startswith('$IAOMIX'):
                i += 1
                if i < len(lines):
                    obj.iaomix = float(lines[i].strip())
            elif line.startswith('$IAOREF'):
                i += 1
                if i < len(lines):
                    obj.iaoref = lines[i].strip()
            elif line.startswith('$IAOPOL'):
                i += 1
                if i < len(lines):
                    obj.iaopol = lines[i].strip()
            elif line.startswith('$HEAVY_ONLY'):
                i += 1
                if i < len(lines):
                    val = lines[i].strip().upper()
                    obj.heavy_only = True if val in ('TRUE', 'YES', '1') else False
            elif line.startswith('$FULL_BASIS'):
                i += 1
                if i < len(lines):
                    val = lines[i].strip().upper()
                    obj.full_basis = True if val in ('TRUE', 'YES', '1') else False
            elif line.startswith('$FRAGMENTS'):
                obj.fragments = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('$'):
                    obj.fragments.append(set(map(int, lines[i].split())))
                    i += 1
                i -= 1
            elif line.startswith('$PARTITION'):
                obj.partition = []
                i += 1
                while i < len(lines) and not lines[i].startswith('$'):
                    p = lines[i].strip()
                    if not p:
                        i += 1
                        continue
                    pup = p.upper()
                    
                    all_fpiaos = [f"fpiao({x})" for x in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]]
                    all_dfpiaos = [f"dfpiao({x})" for x in [0.5, 0.6, 0.7, 0.8, 0.9]]
                    all_peiaos = ["peiao"]
                    all_dpeiaos = [f"dpeiao({x})" for x in [0.5, 0.6, 0.7, 0.8, 0.9]]
                    all_effaos = ["iao-autosad", "iao-effao-gross", "iao-effao-lowdin", "iao-effao-ml", "iao-effao-nao", "wiao"]

                    if pup == 'ALL':
                        obj.partition.extend(['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao'])
                    elif pup == 'ROBUST':
                        obj.partition.extend(['meta_lowdin', 'nao', 'iao'])
                    elif pup == "ALLWIP" or pup == "WIPALL":
                        obj.partition.extend(['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao', 'iao-autosad'])
                        obj.partition.extend([x for x in all_effaos if x not in ['iao-effao-nao', 'iao-autosad']])
                        obj.partition.extend(all_fpiaos)
                        obj.partition.extend(all_dfpiaos)
                        obj.partition.append("peiao")
                        obj.partition.extend(all_dpeiaos)
                        obj.partition.append('iao-effao-nao')
                    elif pup == "ALLWIPNAO":
                        obj.partition.extend(['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao', 'iao-effao-nao'])
                        obj.partition.extend([f"fpiao({x}) nao" for x in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]])
                        obj.partition.extend([f"dfpiao({x}) nao" for x in [0.5, 0.6, 0.7, 0.8, 0.9]])
                        obj.partition.append("peiao nao")
                        obj.partition.extend([f"dpeiao({x}) nao" for x in [0.5, 0.6, 0.7, 0.8, 0.9]])
                    else:
                        obj.partition.append(p)
                    i += 1
                i -= 1
        i += 1
    return obj
