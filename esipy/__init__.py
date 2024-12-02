from os import environ
from esipy.aromaticity import aromaticity
from esipy.make_aoms import make_aoms
from esipy.atomicfiles import write_aoms, read_aoms
from esipy.tools import mol_info, format_partition, load_file, format_short_partition, wf_type


class ESI:
    def __init__(self, Smo=None, rings=None, mol=None, mf=None, myhf = None, partition=None,
                 mci=None, av1245=None, flurefs=None, homarefs=None,
                 homerrefs=None, connectivity=None, geom=None, molinfo=None,
                 ncores=1, saveaoms=None, savemolinfo=None, name="calc", readpath='.'):
        # For usual ESIpy calculations
        self._Smo = Smo
        self.rings = rings
        self.mol = mol
        self.mf = mf
        self.myhf = myhf
        self._partition = partition
        self._mci = mci
        self._av1245 = av1245
        # For custom references
        self.flurefs = flurefs
        self.homarefs = homarefs
        self.homerrefs = homerrefs
        self.connectivity = connectivity
        self.geom = geom
        # For other tools
        self._molinfo = molinfo
        self.name = name
        self.ncores = ncores
        self.saveaoms = saveaoms
        self.savemolinfo = savemolinfo
        self.readpath = readpath

        print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")
        print(" ** Localization & Delocalization Indices **  ")
        print(" ** For Hilbert-Space Atomic Partitioning **  ")
        print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")
        print("   Application to Aromaticity Calculations   ")
        print("  Joan Grebol, Eduard Matito, Pedro Salvador  ")
        print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")

    def aoms(self):
        """
        Generates the Atomic Overlap Matrices (AOMs) in the Molecular Orbitals basis.
        Can be saved in binary through the 'ESI.saveaoms' attribute set to a string.
        Will automatically overwrite 'ESI.Smo' attribute.

        Returns:
            The AOMs in MO basis
        """
        if isinstance(self.partition, list):
            raise ValueError(" | Only one partition at a time. Partition should be a string, not a list.\n | Please consider looping through the partitions before calling the function")

        if self.mol and self.mf and self.partition:
            self._Smo = make_aoms(self.mol, self.mf, partition=self.partition, save=self.saveaoms, myhf=self.myhf)
            if self.saveaoms:
                print(f" | Saved the AOMs in the {self.saveaoms} file")
            return self._Smo
        else:
            raise ValueError(" | Missing variables 'mol', 'mf', or 'partition'")

    @property
    def Smo(self):
        if isinstance(self._Smo, str):
            return load_file(self._Smo)
        if self._Smo is None:
            self._Smo = self.aoms()
        return self._Smo

    @property
    def molinfo(self):
        if isinstance(self._molinfo, str):
            return load_file(self._molinfo)
        if self._molinfo is None:
            self._molinfo = mol_info(self.mol, self.mf, self.savemolinfo, self._partition)
            if self.savemolinfo:
                print(f" | Saved the molinfo in the {self.savemolinfo} file")
        return self._molinfo

    @property
    def partition(self):
        if isinstance(self._partition, str):
            return format_partition(self._partition)
        raise ValueError(" | Partition could not be processed. Options are 'mulliken', 'lowdin', 'meta_lowdin', 'nao' and 'iao'")

    @property
    def mci(self):
        if self._mci is not None:
            return self._mci
        if isinstance(self.rings[0], list): # Check if there are more than one rings
            maxring = max(len(ring) for ring in self.rings)
        else:
            maxring = len(self.rings)
        return maxring < 12

    @mci.setter
    def mci(self, value):
        self._mci = value

    @property
    def av1245(self):
        if self._av1245 is not None:
            return self._av1245
        if isinstance(self.rings[0], list): # Check if there are more than one rings
            maxring = max(len(ring) for ring in self.rings)
        else:
            maxring = len(self.rings)
        return maxring > 9

    @av1245.setter
    def av1245(self, value):
        self._av1245 = value

    def readaoms(self):
        """
        Reads the AOMs from a directory in AIMAll and ESIpy format.
        Overwrites 'ESI.Smo' variable.

        Returns:
            The AOMs in the MO basis.
        """
        if self.readpath is None:
            print(" | No path specified in 'ESI.readpath'. Will assume working directory")
        self._Smo = read_aoms(path=self.readpath)
        print(f" | Read the AOMs from {self.readpath}/{self.name}.aoms")
        return self._Smo

    def writeaoms(self):
        """
        Writes ESIpy's AOMs in AIMAll format.

        Generates:
            - A '_atomicfiles/' directory with all the files created.
            - A '.int' file for each atom with its corresponding AOM.
            - A 'name.files' with a list of the names of the '.int' files.
            - A 'name.bad' with a standard input for the ESI-3D code.
        """
        for attr in ['mol', 'mf', 'name', 'Smo', 'partition']:
            if getattr(self, attr, None) is None:
                raise AttributeError(
                    f" | Missing required attribute '{attr}'. Please define it before calling ESI.writeaoms")

        write_aoms(self.mol, self.mf, self.name, self.Smo, self.rings, self.partition)
        shortpart = format_short_partition(self.partition)
        print(f" | Written the AOMs in {self.readpath}/{self.name}_{shortpart}/")

    def calc(self):
        """
        Main ESIpy. Calculation of population analysis, localization and delocalization indices and
        electronic (and geometric) aromaticity indicators.
        """
        if isinstance(self.partition, list):
            raise ValueError(" | Only one partition at a time. Partition should be a string, not a list\n | Please consider looping through the partitions before calling the function")

        if self.rings is None:
            raise ValueError(" | The variable 'rings' is mandatory and must be a list with the ring connectivity")

        if self._Smo is None:
            print(f" | Partition {self.partition} does not have Smo, generating it")
            self._Smo = self.Smo
            if self._Smo is None:
                raise ValueError(" | Could not build the AOMs from the given data")

        aromaticity(
            Smo=self.Smo,
            rings=self.rings,
            mol=self.mol,
            mf=self.mf,
            partition=self.partition,
            mci=self.mci,
            av1245=self.av1245,
            flurefs=self.flurefs,
            homarefs=self.homarefs,
            homerrefs=self.homerrefs,
            geom=self.geom,
            ncores=self.ncores,
            molinfo=self.molinfo
           )

environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"
environ["MKL_NUM_THREADS"] = "1"


