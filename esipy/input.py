"""
Input parser for ESIpy custom input blocks.
Supports keywords: $READFCHK, $RING, $PARTITION, $FLUREF, $HOMAREF, $FINDRINGS, $MINLEN, $MAXLEN
"""

class ESIInput:
    def __init__(self):
        self.fchk_file = None
        # None means "not specified in input"; empty list means user provided an empty block
        self.rings = None
        # If set True by explicit $NORING, user requested no ring finding
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
        # number of cores requested by the input (None if not provided)
        self.ncores = None
        self.mciaprox = []

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
                        pup = p.upper()
                        if pup == 'ALL':
                            # Expand ALL to all available partitions (lowercase tokens)
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
                # Read subsequent lines until next keyword (starts with $) or EOF
                while i < len(lines) and not lines[i].startswith('$'):
                    # split the line into tokens and try to parse floats (allow scientific notation)
                    for tok in lines[i].split():
                        try:
                            obj.fluref.append(float(tok))
                        except Exception:
                            # ignore tokens that cannot be parsed as float
                            pass
                    i += 1
                i -= 1
            elif line.startswith('$HOMAREF'):
                i += 1
                obj.homaref = []
                # Read subsequent lines until next keyword (starts with $) or EOF
                while i < len(lines) and not lines[i].startswith('$'):
                    for tok in lines[i].split():
                        try:
                            obj.homaref.append(float(tok))
                        except Exception:
                            pass
                    i += 1
                i -= 1
            elif line.startswith('$FINDRINGS'):
                obj.findrings = True
            elif line.startswith('$NORING'):
                # Explicit request: do not find rings and leave rings as None
                obj.noring = True
                obj.findrings = False
                obj.rings = None
            elif line.startswith('$MINLEN'):
                i += 1
                obj.minlen = int(lines[i])
            elif line.startswith('$MAXLEN'):
                i += 1
                obj.maxlen = int(lines[i])
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
                        # leave as None if parsing fails
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
                    n_atoms = int(lines[i])
                    i += 1
                    atoms = set(map(int, lines[i].split()))
                    obj.fragments.append(atoms)
                    i += 1
                i -= 1
            i += 1

        # Finalize save name if $SAVE was set but molecule name not yet determined
        # Set sensible defaults before finalizing save name.
        obj._finalize_defaults()
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
                # No valid name found
                pass

    def _finalize_defaults(self):
        """Apply light defaults when some input blocks are omitted.

        - If no partition lines were provided, default to the robust set:
          meta_lowdin, nao and iao.
        - If no explicit rings were provided and neither $FINDRINGS nor $NORING
          were set by the user, enable automatic ring finding by default.
        """
        # No partition provided. Will use robust by default
        if self.partition is None or len(self.partition) == 0:
            self.partition = ['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao']

        # Normalize partition tokens to lowercase to ensure consistent filename generation
        from esipy.tools import format_partition
        self.partition = [format_partition(p) for p in self.partition]

        # No rings provided. Will find rings using default settings (min=6, max=12)
        # Only enable automatic ring finding when the user did not explicitly
        # request no ring finding ($NORING) and did not request findrings.
        if self.rings is None and not self.findrings and not self.noring:
            self.findrings = True
            self.rings = []

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
