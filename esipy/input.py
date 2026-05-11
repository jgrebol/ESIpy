import os
"""
Input parser for ESIpy custom input blocks.
Supports keywords: $READFCHK, $RING, $PARTITION, $FLUREF, $FINDRINGS, $AV1245, $MINLEN, $MAXLEN
"""


class ESIInput:
    def __init__(self):
        self.fchk_file = None
        self.rings = None
        self.noring = False
        self.partition = None
        self.fragments = None
        self.fluref = []
        self.homaref = []
        self.findrings = False
        self.minlen = None
        self.maxlen = None
        self.domci = True
        self.mci = None
        self.av1245 = None
        self.save = False
        self.writeaoms = False
        # input mode: 'fchk' (default), 'readint', 'readaoms'
        self.mode = 'fchk'
        # for readint: directory containing .int files
        self.readpath = None
        # for readaoms: base name (without extension) to construct aoms/molinfo per partition
        self.aomname = None
        self.ncores = None
        self.mciaprox = []
        self.exclude = []
        self.iaomix = None
        self.iaoref = 'minao'
        self.iaopol = 'ano'
        self.heavy_only = True
        self.full_basis = False

    @staticmethod
    def from_string(input_str):
        obj = ESIInput()
        lines = [line.strip() for line in input_str.strip().splitlines() if line.strip()]

        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('$READFCHK'):
                obj.mode = 'fchk'
                i += 1
                if i < len(lines):
                    obj.fchk_file = lines[i]
            elif line.startswith('$READINT'):
                obj.mode = 'readint'
                i += 1
                if i < len(lines):
                    obj.readpath = lines[i]
            elif line.startswith('$READAOM'):
                obj.mode = 'readaoms'
                i += 1
                if i < len(lines):
                    obj.aomname = lines[i]
            elif line.startswith('$IAOREF'):
                i += 1
                if i < len(lines) and not lines[i].startswith('$'):
                    obj.iaoref = lines[i]
                else:
                    obj.iaoref = 'minao'
                    i -= 1
            elif line.startswith('$IAOPOL'):
                i += 1
                if i < len(lines) and not lines[i].startswith('$'):
                    obj.iaopol = lines[i]
                else:
                    obj.iaopol = 'ano'
                    i -= 1
            elif line.startswith('$HNOPOL'):
                obj.heavy_only = True
            elif line.startswith('$HPOL'):
                obj.heavy_only = False
            elif line.startswith('$FULLBASIS'):
                obj.full_basis = True
            elif line.startswith('$RING') or line.startswith('$RINGS'):
                obj.rings = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('$'):
                    parts = lines[i].split()
                    ring = []
                    for p in parts:
                        if p.upper().startswith('F'):
                            ring.append(p.upper())
                        else:
                            try:
                                ring.append(int(p))
                            except ValueError:
                                ring.append(p)
                    obj.rings.append(ring)
                    i += 1
                i -= 1
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
                    pup = p.upper()
                    
                    all_fpiaos = [f"fpiao({x})" for x in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]]
                    all_dfpiaos = [f"dfpiao({x})" for x in [0.5, 0.6, 0.7, 0.8, 0.9]]
                    all_effaos = ["iao-autosad", "iao-effao-net", "iao-effao-gross", "iao-effao-lowdin", "iao-effao-ml", "iao-effao-nao", "iao-effao-symmetric", "iao-effao-sps", "iao-effao-spsa"]

                    if pup == 'ALL':
                        obj.partition.extend(['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao'])
                    elif pup == 'ROBUST':
                        obj.partition.extend(['meta_lowdin', 'nao', 'iao'])
                    elif pup == "ALLWIP" or pup == "WIPALL":
                        obj.partition.extend(['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao', 'iao-autosad'])
                        obj.partition.extend([x for x in all_effaos if x not in ['iao-effao-nao', 'iao-autosad']])
                        obj.partition.extend([f"fpiao({x})" for x in [1.5, 1.75, 2.0]])
                        obj.partition.extend([f"dfpiao({x})" for x in [0.6, 0.7]])
                        obj.partition.append('iao-effao-nao')
                        obj.partition.extend([f"fpiao({x}) nao" for x in [1.5, 1.75, 2.0]])
                        obj.partition.extend([f"dfpiao({x}) nao" for x in [0.6, 0.7]])
                    elif pup == "ALLWIPNAO":
                        obj.partition.extend(['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao-effao-nao'])
                        obj.partition.extend([f"fpiao({x}) nao" for x in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]])
                        obj.partition.extend([f"dfpiao({x}) nao" for x in [0.5, 0.6, 0.7, 0.8, 0.9]])
                    elif pup == "ALLEDU":
                        obj.partition.append("iao")
                        obj.partition.extend(all_fpiaos)
                        obj.partition.extend(all_dfpiaos)
                    elif pup == "ALLEFFAO" or pup == "ALLEFAO":
                        obj.partition.append("iao")
                        obj.partition.extend(all_effaos)
                    elif pup == "ALLIAO":
                        obj.partition.append("iao")
                        obj.partition.append("iao-autosad")
                        obj.partition.extend(all_effaos)
                        obj.partition.append("iao_basis")
                        obj.partition.extend(all_fpiaos)
                        obj.partition.extend(all_dfpiaos)
                    else:
                        obj.partition.append(p)
                    i += 1
                i -= 1
            elif line.upper().startswith('$ALLPARTS'):
                obj.partition = ['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao', 'iao-autosad', 'iao-effao', 'iao-effao-lowdin']
            elif line.startswith('$FLUREF'):
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
            elif line.startswith('$FINDRING'):
                obj.findrings = True
            elif line.startswith('$AV1245'):
                obj.av1245 = True
            elif line.startswith('$NORING'):
                obj.noring = True
                obj.findrings = False
                obj.rings = None
            elif line.strip().startswith('$NOMCI'):
                obj.domci = False
                obj.mci = False
            elif line.strip().startswith('$MCI') or line.strip().startswith('$DOMCI'):
                obj.domci = True
                obj.mci = True
            elif line.startswith('$MINLEN'):
                i += 1
                obj.minlen = int(lines[i])
            elif line.startswith('$MAXLEN'):
                i += 1
                obj.maxlen = int(lines[i])
            elif line.startswith('$NCORES') or line.startswith('$NCORE'):
                i += 1
                obj.ncores = int(lines[i])
            elif line.startswith('$EXCLUDE'):
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
            i += 1
        return obj

    @staticmethod
    def from_file(filepath):
        with open(filepath, 'r') as f:
            return ESIInput.from_string(f.read())
