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

        # Trackers for contradictory keywords
        seen_modes = []
        seen_ring_cmds = []
        seen_mci_cmds = []

        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('$READFCHK'):
                seen_modes.append('$READFCHK')
                obj.mode = 'fchk'
                i += 1
                if i < len(lines):
                    obj.fchk_file = lines[i]
            elif line.startswith('$READINT'):
                seen_modes.append('$READINT')
                obj.mode = 'readint'
                i += 1
                if i < len(lines):
                    obj.readpath = lines[i]
            elif line.startswith('$READAOM'):
                seen_modes.append('$READAOM')
                obj.mode = 'readaoms'
                i += 1
                if i < len(lines):
                    obj.aomname = lines[i]
            elif line.startswith('$RING'):
                seen_ring_cmds.append('$RING')
                obj.rings = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('$'):
                    obj.rings.append(list(map(int, lines[i].split())))
                    i += 1
                i -= 1
            elif line.startswith('$PARTITION'):
                obj.partition = []
                i += 1
                while i < len(lines) and not lines[i].startswith('$'):
                    partitions = lines[i].split()
                    for p in partitions:
                        pup = p.upper()
                        if pup == 'ALL':
                            obj.partition.extend(
                                ['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao', 'iao-autosad', 'iao-effao',
                                 'iao-effao-lowdin'])
                        elif pup == 'ROBUST':
                            obj.partition.extend(['meta_lowdin', 'nao', 'iao'])
                        else:
                            obj.partition.append(p.strip().lower())
                    i += 1
                i -= 1
            elif line.upper().startswith('$ALLPARTS') or line.upper().startswith('$ALLPARTITIONS'):
                obj.partition = ['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao', 'iao-autosad', 'iao-effao',
                                 'iao-effao-lowdin']
            elif line.upper().startswith('$AUTO'):
                seen_ring_cmds.append('$AUTO')
                obj.partition = ['mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao', 'iao-autosad', 'iao-effao',
                                 'iao-effao-lowdin']
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
            elif line.startswith('$FINDRINGS'):
                seen_ring_cmds.append('$FINDRINGS')
                obj.findrings = True
            elif line.startswith('$NORING'):
                seen_ring_cmds.append('$NORING')
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
                            obj.exclude.append(int(p))
                        except ValueError:
                            obj.exclude.append(p)
                    i += 1
                i -= 1
            elif line.startswith('$MCIALG') or line.startswith('$MCIAPROX'):
                seen_mci_cmds.append('$MCIAPROX')
                i += 1
                while i < len(lines) and not lines[i].startswith('$'):
                    parts = lines[i].split()
                    if len(parts) >= 2:
                        try:
                            alg = int(parts[0])
                            dist = int(parts[1])
                            obj.mciaprox.append((alg, dist))
                        except ValueError:
                            pass
                    i += 1
                i -= 1
            elif line.startswith('$MCI'):
                seen_mci_cmds.append('$MCI')
                obj.mci = True
            elif line.startswith('$NOMCI'):
                seen_mci_cmds.append('$NOMCI')
                obj.mci = False
            elif line.startswith('$AV1245'):
                obj.av1245 = True
            elif line.startswith('$NCORES'):
                i += 1
                if i < len(lines):
                    try:
                        obj.ncores = int(lines[i])
                    except Exception:
                        obj.ncores = None
            elif line.startswith('$SAVE'):
                obj.save = True
                if obj.mode == 'fchk' and obj.fchk_file:
                    import os
                    obj.save = os.path.splitext(os.path.basename(obj.fchk_file))[0]
                elif obj.mode == 'readint' and obj.readpath:
                    import os
                    obj.save = os.path.basename(os.path.normpath(obj.readpath))
                elif obj.mode == 'readaoms' and obj.aomname:
                    obj.save = obj.aomname
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

        # --- Validation Checks ---
        unique_modes = set(seen_modes)
        if len(unique_modes) > 1:
            raise ValueError(
                f"Contradictory input modes specified: {', '.join(unique_modes)}. Please specify only one input source.")

        unique_ring_cmds = set(seen_ring_cmds)
        if '$NORING' in unique_ring_cmds and len(unique_ring_cmds) > 1:
            conflicts = [c for c in unique_ring_cmds if c != '$NORING']
            raise ValueError(
                f"Contradictory ring instructions specified. Found $NORING along with: {', '.join(conflicts)}.")

        unique_mci_cmds = set(seen_mci_cmds)
        if '$NOMCI' in unique_mci_cmds and len(unique_mci_cmds) > 1:
            conflicts = [c for c in unique_mci_cmds if c != '$NOMCI']
            raise ValueError(
                f"Contradictory MCI instructions specified. Found $NOMCI along with: {', '.join(conflicts)}.")
        # -------------------------

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
                pass

    def _set_defaults(self):
        """Apply light defaults when some input blocks are omitted."""
        if self.partition is None or len(self.partition) == 0:
            raise ValueError(
                'No partition specified in input. Please set $PARTITION to one or more of: '
                "'mulliken', 'lowdin', 'meta_lowdin', 'nao', 'iao' (or use 'ALL'/'ROBUST')."
            )

        from esipy.tools import format_partition
        self.partition = [format_partition(p) for p in self.partition]

    @staticmethod
    def from_file(filepath):
        with open(filepath) as f:
            return ESIInput.from_string(f.read())

    def __repr__(self):
        return (
            f"ESIInput(mode={self.mode}, fchk_file={self.fchk_file}, readpath={self.readpath}, aomname={self.aomname}, "
            f"rings={self.rings}, noring={self.noring}, partition={self.partition}, fragments={self.fragments}, "
            f"fluref={self.fluref}, homaref={self.homaref}, findrings={self.findrings}, "
            f"minlen={self.minlen}, maxlen={self.maxlen}, mci={self.mci}, av1245={self.av1245}, "
            f"save={self.save}, writeaoms={self.writeaoms}, ncores={self.ncores})")
