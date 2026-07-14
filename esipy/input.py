import os
"""
Input parser for ESIpy custom input blocks.
Supports keywords: $READFCHK, $RING, $PARTITION, $FLUREF, $FINDRINGS, $AV1245, $MINLEN, $MAXLEN
"""


class ESIInput:
    def __init__(self):
        self.fchk_file = None
        self.ecp = None
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
            elif line.startswith('$ECP'):
                parts = line.split()
                if len(parts) > 1:
                    obj.ecp = parts[1].strip()
                else:
                    i += 1
                    if i < len(lines): obj.ecp = lines[i].strip()
            elif line.startswith('$READAOM'):
                obj.mode = 'readaoms'
                i += 1
                if i < len(lines):
                    obj.aomname = lines[i]

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
                    partitions = lines[i].split()
                    for p in partitions:
                        pup = p.upper()
                        if pup in ('ALL', 'ROBUST', 'ALLWIP', 'WIPALL', 'ALLEDU', 'ALLEFFAO', 'ALLEFAO', 'ALLIAO'):
                            obj.partition.extend(['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao'])
                        else:
                            obj.partition.append(p.strip().lower())
                    i += 1
                i -= 1
            elif line.upper().startswith('$ALLPARTS') or line.upper().startswith('$ALLPARTITIONS'):
                obj.partition = ['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao']
            elif line.upper().startswith('$AUTO') or line.upper().startswith('$DEFAULT'):
                seen_ring_cmds.append('$AUTO')
                obj.partition = ['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao']
                obj.findrings = True
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
