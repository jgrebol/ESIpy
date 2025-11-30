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
        self.domci = True
        self.doav1245 = False
        self.save = False
        self.writeaoms = False
        # input mode: 'fchk' (default), 'readint', 'readaoms'
        self.mode = 'fchk'
        # for readint: directory containing .int files
        self.readpath = None
        # for readaoms: base name (without extension) to construct aoms/molinfo per partition
        self.aomname = None

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
                # next line: path to directory with .int files
                obj.mode = 'readint'
                i += 1
                if i < len(lines):
                    obj.readpath = lines[i]
            elif line.startswith('$READAOMS'):
                # next line: base name (without extension) used to get .aoms and .molinfo
                obj.mode = 'readaoms'
                i += 1
                if i < len(lines):
                    obj.aomname = lines[i]
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
                    partitions = lines[i].split()
                    for p in partitions:
                        if p.upper() == 'ALL':
                            # Expand ALL to all available partitions
                            obj.partition.extend(['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao'])
                        elif p.upper() == 'ROBUST':
                            # Expand ROBUST to robust partitions
                            obj.partition.extend(['meta_lowdin', 'nao', 'iao'])
                        else:
                            obj.partition.append(p)
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
                obj.domci = False
            elif line.startswith('$AV1245'):
                obj.doav1245 = True
            elif line.startswith('$SAVE'):
                obj.save = True
                # Determine molecule name based on input mode
                # This will be used to create subdirectory and save files
                if obj.mode == 'fchk' and obj.fchk_file:
                    # Extract filename without extension
                    import os
                    obj.save = os.path.splitext(os.path.basename(obj.fchk_file))[0]
                elif obj.mode == 'readint' and obj.readpath:
                    # Use directory basename
                    import os
                    obj.save = os.path.basename(os.path.normpath(obj.readpath))
                elif obj.mode == 'readaoms' and obj.aomname:
                    # Use the aomname directly
                    obj.save = obj.aomname
                # If mode/file not set yet, keep as True (will be resolved later)
            elif line.startswith('$WRITEAOMS'):
                obj.writeaoms = True
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

        # Finalize save name if $SAVE was set but molecule name not yet determined
        obj._finalize_save_name()
        return obj

    def _finalize_save_name(self):
        """Finalize save name based on input mode if it's still just True."""
        if self.save is True:
            import os
            if self.mode == 'fchk' and self.fchk_file:
                self.save = os.path.splitext(os.path.basename(self.fchk_file))[0]
            elif self.mode == 'readint' and self.readpath:
                self.save = os.path.basename(os.path.normpath(self.readpath))
            elif self.mode == 'readaoms' and self.aomname:
                self.save = self.aomname
            else:
                # No valid name found, keep as True (will be handled by caller)
                pass

    @staticmethod
    def from_file(filepath):
        with open(filepath) as f:
            return ESIInput.from_string(f.read())

    def __repr__(self):
        return (f"ESIInput(mode={self.mode}, fchk_file={self.fchk_file}, readpath={self.readpath}, aomname={self.aomname}, "
                f"rings={self.rings}, partition={self.partition}, fragments={self.fragments}, "
                f"fluref={self.fluref}, homaref={self.homaref}, findrings={self.findrings}, "
                f"minlen={self.minlen}, maxlen={self.maxlen}, domci={self.domci}, doav1245={self.doav1245}, "
                f"save={self.save}, writeaoms={self.writeaoms})")
