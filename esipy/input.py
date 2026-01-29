"""
Input parser for ESIpy custom input blocks.
Supports keywords: $READFCHK, $RING, $PARTITION, $FLUREF, $FINDRINGS, $MINLEN, $MAXLEN
"""

class ESIInput:
    def __init__(self):
        self.fchk_file = None
        self.rings = None
        self.noring = False
        self.partition = None
        self.fragments = []  # List of sets
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
                # next line: name of the .fchk file
                obj.mode = 'fchk'
                i += 1
                if i < len(lines):
                    obj.fchk_file = lines[i]
            elif line.startswith('$READINT'): # $READINTS also works
                # next line: base name of the directory with the int files
                obj.mode = 'readint'
                i += 1
                if i < len(lines):
                    obj.readpath = lines[i]
            elif line.startswith('$READAOM'):
                # next line: base name (without extension) used to get .aoms and .molinfo
                obj.mode = 'readaoms'
                i += 1
                if i < len(lines):
                    obj.aomname = lines[i]
            elif line.startswith('$RING'): # $RINGS also works
                obj.rings = []
                i += 1
                # Keep reading lines as long as they exist and don't start with '$'
                while i < len(lines) and not lines[i].strip().startswith('$'):
                    obj.rings.append(list(map(int, lines[i].split())))
                    i += 1
                i -= 1
            elif line.startswith('$PARTITION'): # $PARTITIONS also works
                obj.partition = []
                i += 1
                while i < len(lines) and not lines[i].startswith('$'):
                    partitions = lines[i].split()
                    for p in partitions:
                        pup = p.upper()
                        if pup == 'ALL':
                            # Expand ALL to all available partitions
                            obj.partition.extend(['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao'])
                        elif pup == 'ROBUST':
                            # Expand ROBUST to robust partitions
                            obj.partition.extend(['meta_lowdin', 'nao', 'iao'])
                        else:
                            # normalize token to lowercase for consistent filename generation
                            obj.partition.append(p.strip().lower())
                    i += 1
                i -= 1
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
            elif line.startswith('$FINDRINGS'):
                obj.findrings = True
            elif line.startswith('$NORING'):
                # No ring analysis, only atomic populations and delocalization indices
                obj.noring = True
                obj.findrings = False
                obj.rings = None
            elif line.startswith('$MINLEN'):
                i += 1
                obj.minlen = int(lines[i])
            elif line.startswith('$MAXLEN'):
                i += 1
                obj.maxlen = int(lines[i])
            elif line.startswith('$EXCLUDE'):
                obj.exclude = []
                i += 1
                while i < len(lines) and not lines[i].startswith('$'):
                    parts = lines[i].split()
                    for p in parts:
                        try:
                            # Try to convert to int (atom index)
                            obj.exclude.append(int(p))
                        except ValueError:
                            # If it fails, treat as element symbol (str)
                            obj.exclude.append(p)
                    i += 1
                i -= 1
            elif line.startswith('$MCIALG'):
                i += 1
                while i < len(lines) and not lines[i].startswith('$'):
                    parts = lines[i].split()
                    if len(parts) >= 2:
                        try:
                            # Format: Algorithm Distance
                            alg = int(parts[0])
                            dist = int(parts[1])
                            obj.mciaprox.append((alg, dist))
                        except ValueError:
                            pass  # Skip malformed lines
                    i += 1
                i -= 1
            elif line.startswith('$MCI'):
                obj.mci = True
            elif line.startswith('$NOMCI'):
                obj.mci = False
            elif line.startswith('$AV1245'):
                obj.av1245 = True
            elif line.startswith('$NCORES'):
                i += 1
                if i < len(lines):
                    try:
                        obj.ncores = int(lines[i])
                    except Exception:
                        # leave as None if parsing fails. ESI will make it single core then
                        obj.ncores = None
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
                    i += 1
                    atoms = set(map(int, lines[i].split()))
                    obj.fragments.append(atoms)
                    i += 1
                i -= 1
            i += 1

        obj._set_defaults()
        obj._set_save_name()
        return obj

    def _set_save_name(self):
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
                # No valid name found
                pass

    def _set_defaults(self):
        """Apply light defaults when some input blocks are omitted.

        - If no explicit rings were provided and neither $FINDRINGS nor $NORING
          were set by the user, enable automatic ring finding by default.
        """
        # If no partition provided, raise an informative error (user must choose)
        if self.partition is None or len(self.partition) == 0:
            raise ValueError(
                'No partition specified in input. Please set $PARTITION to one or more of: '
                "'mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao' (or use 'ALL'/'ROBUST')."
            )

        # Format partitions
        from esipy.tools import format_partition
        self.partition = [format_partition(p) for p in self.partition]

        return

    @staticmethod
    def from_file(filepath):
        with open(filepath) as f:
            return ESIInput.from_string(f.read())

    def __repr__(self):
        return (f"ESIInput(mode={self.mode}, fchk_file={self.fchk_file}, readpath={self.readpath}, aomname={self.aomname}, "
                f"rings={self.rings}, noring={self.noring}, partition={self.partition}, fragments={self.fragments}, "
                f"fluref={self.fluref}, homaref={self.homaref}, findrings={self.findrings}, "
                f"minlen={self.minlen}, maxlen={self.maxlen}, mci={self.mci}, av1245={self.av1245}, "
                f"save={self.save}, writeaoms={self.writeaoms}, ncores={self.ncores})")
