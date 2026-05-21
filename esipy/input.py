import os
import numpy as np
from pyscf import gto, scf

class ESIInput:
    def __init__(self):
        # Infrastructure from main/dev-iao
        self.fchk_file = None
        self.rings = None
        self.noring = False
        self.partition = []
        self.fragments = []
        self.fluref = []
        self.homaref = []
        self.findrings = False
        self.minlen = None
        self.maxlen = None
        self.domci = True
        self.mci = None
        self.av1245 = None
        self.save = None
        self.writeaoms = False
        self.mode = 'fchk'
        self.readpath = None
        self.aomname = None
        self.ncores = None
        self.mciaprox = []
        self.exclude = []
        # New IAO/other fields from script
        self.iaomix = None
        self.iaoref = None
        self.iaopol = None
        self.heavy_only = False
        self.full_basis = False

    @classmethod
    def from_file(cls, path):
        return read_input(path)

def read_input(path):
    obj = ESIInput()
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
            pup_line = line.upper()
            if pup_line.startswith('$FCHK') or pup_line.startswith('$READFCHK'):
                i += 1
                if i < len(lines):
                    obj.fchk_file = lines[i].strip()
            elif pup_line.startswith('$SAVE'):
                i += 1
                if i < len(lines):
                    obj.save = lines[i].strip()
            elif pup_line.startswith('$FRAGMENTS'):
                obj.fragments = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('$'):
                    obj.fragments.append(set(map(int, lines[i].split())))
                    i += 1
                i -= 1
            elif pup_line.startswith('$PARTITION'):
                obj.partition = []
                i += 1
                while i < len(lines) and not lines[i].startswith('$'):
                    p = lines[i].strip()
                    if not p:
                        i += 1
                        continue
                    pup = p.upper()
                    
                    # Define Families
                    original_family = ['mulliken', 'lowdin', 'meta-lowdin', 'nao', 'iao', 'iao2', 'iao-autosad']
                    effao_family = ['iao-effao-gross', 'iao-effao-net', 'iao-effao-lowdin', 'iao-effao-meta-lowdin', 
                                    'iao-effao-nao', 'iao-effao-symmetric', 'iao-effao-sps', 'iao-effao-spsa']
                    fpiao_family = ['fpiao(1.0)', 'fpiao(1.25)', 'fpiao(1.5)', 'fpiao(1.75)', 'fpiao(2.0)']
                    dfpiao_family = ['dfpiao(0.5)', 'dfpiao(0.6)', 'dfpiao(0.7)', 'dfpiao(0.8)', 'dfpiao(0.9)']
                    peiao_family = ['peiao', 'dpeiao(0.5)', 'dpeiao(0.6)', 'dpeiao(0.7)', 'dpeiao(0.8)', 'dpeiao(0.9)']

                    if pup in ('ALL', 'ROBUST', 'ALLPARTS', 'ALLWIP'):
                        obj.partition.extend(original_family + effao_family + fpiao_family + dfpiao_family + peiao_family)
                    elif pup == 'ALLEFFAO':
                        obj.partition.extend(original_family + effao_family)
                    elif pup == 'ALLFPIAO':
                        obj.partition.extend(original_family + fpiao_family)
                    elif pup == 'ALLDFPIAO':
                        obj.partition.extend(original_family + dfpiao_family)
                    elif pup == 'ALLPEIAO':
                        obj.partition.extend(original_family + peiao_family)
                    else:
                        obj.partition.append(p)
                    i += 1
                i -= 1
            elif pup_line.startswith('$ALLPARTS') or pup_line.startswith('$ALLWIP'):
                original_family = ['mulliken', 'lowdin', 'meta-lowdin', 'nao', 'iao', 'iao2', 'iao-autosad']
                effao_family = ['iao-effao-gross', 'iao-effao-net', 'iao-effao-lowdin', 'iao-effao-meta-lowdin', 
                                'iao-effao-nao', 'iao-effao-symmetric', 'iao-effao-sps', 'iao-effao-spsa']
                fpiao_family = ['fpiao(1.0)', 'fpiao(1.25)', 'fpiao(1.5)', 'fpiao(1.75)', 'fpiao(2.0)']
                dfpiao_family = ['dfpiao(0.5)', 'dfpiao(0.6)', 'dfpiao(0.7)', 'dfpiao(0.8)', 'dfpiao(0.9)']
                peiao_family = ['peiao', 'dpeiao(0.5)', 'dpeiao(0.6)', 'dpeiao(0.7)', 'dpeiao(0.8)', 'dpeiao(0.9)']
                obj.partition = original_family + effao_family + fpiao_family + dfpiao_family + peiao_family
            elif pup_line.startswith('$RING') or pup_line.startswith('$RINGS'):
                obj.rings = []
                i += 1
                while i < len(lines) and not lines[i].startswith('$'):
                    obj.rings.append(list(map(int, lines[i].split())))
                    i += 1
                i -= 1
            elif pup_line.startswith('$FLUREF'):
                i += 1
                obj.fluref = []
                while i < len(lines) and not lines[i].startswith('$'):
                    for tok in lines[i].split():
                        try:
                            obj.fluref.append(float(tok))
                        except Exception:
                            pass
                    i += 1
                i -= 1
            elif pup_line.startswith('$FINDRINGS'):
                obj.findrings = True
            elif pup_line.startswith('$NORING'):
                obj.noring = True
                obj.findrings = False
                obj.rings = None
            elif pup_line.strip().startswith('$NOMCI'):
                obj.domci = False
            elif pup_line.startswith('$MINLEN'):
                i += 1
                obj.minlen = int(lines[i])
            elif pup_line.startswith('$MAXLEN'):
                i += 1
                obj.maxlen = int(lines[i])
            elif pup_line.startswith('$EXCLUDE'):
                obj.exclude = []
                i += 1
                while i < len(lines) and not lines[i].startswith('$'):
                    parts = lines[i].split()
                    for p in parts:
                        try:
                            obj.exclude.append(int(p))
                        except ValueError:
                            obj.exclude.append(p)
                    i += 1
                i -= 1
            elif pup_line.startswith('$NCORES'):
                i += 1
                if i < len(lines):
                    obj.ncores = int(lines[i].strip())
            elif pup_line.startswith('$AV1245'):
                i += 1
                if i < len(lines):
                    obj.av1245 = float(lines[i].strip())
            elif pup_line.startswith('$WRITEAOMS'):
                obj.writeaoms = True
            elif pup_line.startswith('$READINT'):
                obj.mode = 'readint'
                i += 1
                if i < len(lines):
                    obj.readpath = lines[i].strip()
            elif pup_line.startswith('$READAOMS'):
                obj.mode = 'readaoms'
                i += 1
                if i < len(lines):
                    obj.aomname = lines[i].strip()
            elif pup_line.startswith('$IAOMIX'):
                i += 1
                if i < len(lines):
                    obj.iaomix = lines[i].strip()
            elif pup_line.startswith('$IAOREF'):
                i += 1
                if i < len(lines):
                    obj.iaoref = lines[i].strip()
            elif pup_line.startswith('$IAOPOL'):
                i += 1
                if i < len(lines):
                    obj.iaopol = lines[i].strip()
            elif pup_line.startswith('$HEAVYONLY'):
                obj.heavy_only = True
            elif pup_line.startswith('$FULLBASIS'):
                obj.full_basis = True
        i += 1
    return obj

