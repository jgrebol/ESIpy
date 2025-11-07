"""
Input parser for ESIpy custom input blocks.
Supports keywords: $READFCHK, $RING, $PARTITION, $FLUREF, $HOMAREF, $FINDRINGS, $MINLEN, $MAXLEN
"""
import re

class ESIInput:
    def __init__(self):
        self.fchk_file = None
        self.rings = []
        self.partition = []
        self.fragments = []  # List of sets
        self.fluref = []
        self.homaref = []
        self.findrings = False
        self.minlen = None
        self.maxlen = None
        self.nomci = True
        self.noav1245 = False
        self.save = False

    @staticmethod
    def from_string(input_str):
        obj = ESIInput()
        lines = [line.strip() for line in input_str.strip().splitlines() if line.strip()]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('$READFCHK'):
                i += 1
                obj.fchk_file = lines[i]
            elif line.startswith('$RING'):
                i += 1
                n_rings = int(lines[i])
                i += 1
                obj.rings = []
                for _ in range(n_rings):
                    n_atoms = int(lines[i])
                    i += 1
                    atoms = list(map(int, lines[i].split()))
                    obj.rings.append(atoms)
                    i += 1
                i -= 1
            elif line.startswith('$PARTITION'):
                obj.partition = []
                i += 1
                # Collect all partition values until next $KEYWORD or end of lines
                while i < len(lines) and not lines[i].startswith('$'):
                    obj.partition.extend(lines[i].split())
                    i += 1
                i -= 1
            elif line.startswith('$FLUREF'):
                i += 1
                obj.fluref = []
                while i < len(lines) and re.match(r'^[-+]?[0-9]*\.?[0-9]+$', lines[i]):
                    obj.fluref.append(float(lines[i]))
                    i += 1
                i -= 1
            elif line.startswith('$HOMAREF'):
                i += 1
                obj.homaref = []
                while i < len(lines) and re.match(r'^[-+]?[0-9]*\.?[0-9]+$', lines[i]):
                    obj.homaref.append(float(lines[i]))
                    i += 1
                i -= 1
            elif line.startswith('$FINDRINGS'):
                obj.findrings = True
            elif line.startswith('$MINLEN'):
                i += 1
                obj.minlen = int(lines[i])
            elif line.startswith('$MAXLEN'):
                i += 1
                obj.maxlen = int(lines[i])
            elif line.startswith('$NOMCI'):
                obj.nomci = False
            elif line.startswith('$AV1245'):
                obj.noav1245 = True
            elif line.startswith('$SAVE'):
                obj.save = True
            elif line.startswith('$FRAGMENTS'):
                obj.fragments = []
                i += 1
                n_frag = int(lines[i])
                i += 1
                for _ in range(n_frag):
                    n_atoms = int(lines[i])
                    i += 1
                    atoms = set(map(int, lines[i].split()))
                    obj.fragments.append(atoms)
                    i += 1
                i -= 1
            i += 1
        return obj

    @staticmethod
    def from_file(filepath):
        with open(filepath) as f:
            return ESIInput.from_string(f.read())

    def __repr__(self):
        return (f"ESIInput(fchk_file={self.fchk_file}, rings={self.rings}, partition={self.partition}, fragments={self.fragments}, "
                f"fluref={self.fluref}, homaref={self.homaref}, findrings={self.findrings}, "
                f"minlen={self.minlen}, maxlen={self.maxlen}, mci={self.mci}, av1245={self.av1245}, save={self.save})")
