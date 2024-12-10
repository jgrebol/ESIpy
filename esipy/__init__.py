from os import environ
import numpy as np
from esipy.make_aoms import make_aoms
from esipy.atomicfiles import write_aoms, read_aoms
from esipy.tools import mol_info, format_partition, load_file, format_short_partition, wf_type
from esipy.indicators import (
    compute_iring, sequential_mci, multiprocessing_mci, compute_av1245, compute_pdi,
    compute_flu, compute_boa, compute_homer, compute_homa, compute_bla,
    compute_iring_no, sequential_mci_no, multiprocessing_mci_no, compute_av1245_no,
    compute_pdi_no, compute_boa_no
)

class IndicatorsRest:
    def __init__(self, Smo=None, rings=None, mol=None, mf=None, myhf=None, partition=None, mci=None, av1245=None, flurefs=None, homarefs=None, homerrefs=None, connectivity=None, geom=None, molinfo=None, ncores=1):
        """
                Initialize the indicators from Restricted calculations.

                Parameters:
                Smo (optional): Atomic Overlap Matrices (AOMs) in the MO basis.
                rings (optional): The rings parameter.
                mol (optional): The mol parameter.
                mf (optional): The mf parameter.
                myhf (optional): The myhf parameter.
                partition (optional): The partition parameter.
                mci (optional): The mci parameter.
                av1245 (optional): The av1245 parameter.
                flurefs (optional): The flurefs parameter.
                homarefs (optional): The homarefs parameter.
                homerrefs (optional): The homerrefs parameter.
                connectivity (optional): The connectivity parameter.
                geom (optional): The geom parameter.
                molinfo (optional): The molinfo parameter.
                ncores (int, optional): The number of cores to use. Default is 1.
                """
        self._Smo = Smo
        self._rings = rings
        self._mol = mol
        self._mf = mf
        self._myhf = myhf
        self._partition = partition
        self._mci = mci
        self._av1245 = av1245
        self._flurefs = flurefs
        self._homarefs = homarefs
        self._homerrefs = homerrefs
        self._connectivity = connectivity
        self._geom = geom
        self._molinfo = molinfo
        self._ncores = ncores

    @property
    def iring(self):
        return 2 * compute_iring(self._rings, self._Smo)

    @property
    def mci(self):
        if not hasattr(self, '_done_mci'):
            if self._ncores == 1:
                self._done_mci = sequential_mci(self._rings, self._Smo, self._partition)
            else:
                self._done_mci = multiprocessing_mci(self._rings, self._Smo, self._ncores, self._partition)
        return 2 * self._done_mci

    def _av(self):
        if not hasattr(self, '_done_av'):
            self._done_av = compute_av1245(self._rings, self._Smo, self._partition)
        return self._done_av

    @property
    def av1245(self):
        return 2 * self._av()[0]

    @property
    def avmin(self):
        return 2 * self._av()[1]

    @property
    def av1245_list(self):
        return 2 * np.array(self._av()[2], dtype=object)

    def _pdi(self):
        if not hasattr(self, '_done_pdi'):
            self._done_pdi = compute_pdi(self._rings, self._Smo)
        return self._done_pdi

    @property
    def pdi(self):
        return 2 * self._pdi()[0]

    @property
    def pdi_list(self):
        return 2 * np.array(self._pdi()[1], dtype=object)

    @property
    def flu(self):
        return compute_flu(self._rings, self._molinfo, self._Smo, self._flurefs, self._partition)

    def _boa(self):
        if not hasattr(self, '_done_boa'):
            self._done_boa = compute_boa(self._rings, self._Smo)
        return self._done_boa

    @property
    def boa(self):
        return 2 * self._boa()[0]

    @property
    def boa_c(self):
        return 2 * self._boa()[1]

    @property
    def homer(self):
        if self._geom is None or self._homerrefs is None or self._connectivity is None:
            return None
        else:
            return compute_homer(self._rings, self._molinfo, self._homerrefs)

    def _homa(self):
        if not hasattr(self, '_done_homa'):
            self._done_homa = compute_homa(self._rings, self._molinfo, self._homarefs)
        return self._done_homa

    @property
    def homa(self):
        if self._homa() is None:
            return None
        return self._homa()[0]

    @property
    def en(self):
        return self._homa()[1]

    @property
    def geo(self):
        return self._homa()[2]

    def _bla(self):
        if not hasattr(self, '_done_bla'):
            self._done_bla = compute_bla(self._rings, self._molinfo)
        return self._done_bla

    @property
    def bla(self):
        return self._bla()[0]

    @property
    def bla_c(self):
        return self._bla()[1]

    @property
    def homer(self):
        return compute_homer(self._rings, self._molinfo, self._homerrefs)

class IndicatorsUnrest:
    """
            Initialize the indicators from Unrestricted calculations.

            Parameters:
            Smo (optional): Atomic Overlap Matrices (AOMs) in the MO basis.
            rings (optional): The rings parameter.
            mol (optional): The mol parameter.
            mf (optional): The mf parameter.
            myhf (optional): The myhf parameter.
            partition (optional): The partition parameter.
            mci (optional): The mci parameter.
            av1245 (optional): The av1245 parameter.
            flurefs (optional): The flurefs parameter.
            homarefs (optional): The homarefs parameter.
            homerrefs (optional): The homerrefs parameter.
            connectivity (optional): The connectivity parameter.
            geom (optional): The geom parameter.
            molinfo (optional): The molinfo parameter.
            ncores (int, optional): The number of cores to use. Default is 1.
            """

    def __init__(self, Smo=None, rings=None, mol=None, mf=None, myhf=None, partition=None, mci=None, av1245=None, flurefs=None, homarefs=None, homerrefs=None, connectivity=None, geom=None, molinfo=None, ncores=1):
        self._Smo = Smo
        self._rings = rings
        self._mol = mol
        self._mf = mf
        self._myhf = myhf
        self._partition = partition
        self._mci = mci
        self._av1245 = av1245
        self._flurefs = flurefs
        self._homarefs = homarefs
        self._homerrefs = homerrefs
        self._connectivity = connectivity
        self._geom = geom
        self._molinfo = molinfo
        self._ncores = ncores

    def _irings(self):
        if not hasattr(self, '_done_irings'):
            self._done_irings = (
                compute_iring(self._rings, self._Smo[0]),
                compute_iring(self._rings, self._Smo[1])
            )
        return self._done_irings

    @property
    def iring(self):
        return self._irings()[0] + self._irings()[1]

    @property
    def iring_alpha(self):
        return self._irings()[0]

    @property
    def iring_beta(self):
        return self._irings()[1]

    def _mcis(self):
        if not hasattr(self, '_done_mcis'):
            if self._ncores == 1:
                mci_alpha = sequential_mci(self._rings, self._Smo[0], self._partition)
                mci_beta = sequential_mci(self._rings, self._Smo[1], self._partition)
            else:
                mci_alpha = multiprocessing_mci(self._rings, self._Smo[0], self._ncores,
                                                                 self._partition)
                mci_beta = multiprocessing_mci(self._rings, self._Smo[1], self._ncores,
                                                                self._partition)
            self._done_mcis = (mci_alpha, mci_beta)
        return self._done_mcis

    @property
    def mci(self):
        return self._mcis()[0] + self._mcis()[1]

    @property
    def mci_alpha(self):
        return self._mcis()[0]

    @property
    def mci_beta(self):
        return self._mcis()[1]

    def _avs(self):
        if not hasattr(self, '_done_avs'):
            self._done_avs = (
                compute_av1245(self._rings, self._Smo[0], self._partition),
                compute_av1245(self._rings, self._Smo[1], self._partition)
            )
        return self._done_avs

    @property
    def av1245(self):
        return self._avs()[0][0] + self._avs()[1][0]

    @property
    def av1245_alpha(self):
        return self._avs()[0][0]

    @property
    def av1245_beta(self):
        return self._avs()[1][0]

    @property
    def avmin(self):
        return min(list(self.av1245_list), key=abs)

    @property
    def avmin_alpha(self):
        return min(self.av1245_list_alpha, key=abs)

    @property
    def avmin_beta(self):
        return min(self.av1245_list_beta, key=abs)

    @property
    def av1245_list(self):
        return np.add(self.av1245_list_alpha, self.av1245_list_beta)

    @property
    def av1245_list_alpha(self):
        return self._avs()[0][2]

    @property
    def av1245_list_beta(self):
        return self._avs()[1][2]

    def _pdis(self):
        if not hasattr(self, '_done_pdis'):
            self._done_pdis = (
                compute_pdi(self._rings, self._Smo[0]),
                compute_pdi(self._rings, self._Smo[1])
            )
        return self._done_pdis

    @property
    def pdi(self):
        return self._pdis()[0][0] + self._pdis()[1][0]

    @property
    def pdi_alpha(self):
        return self._pdis()[0][0]

    @property
    def pdi_beta(self):
        return self._pdis()[1][0]

    @property
    def pdi_list(self):
        return self._pdis()[0][1] + self._pdis()[1][1]

    @property
    def pdi_list_alpha(self):
        return self._pdis()[0][1]

    @property
    def pdi_list_beta(self):
        return self._pdis()[1][1]

    def _flus(self):
        if not hasattr(self, '_done_flus'):
            self._done_flus = (
                compute_flu(self._rings, self._molinfo, self._Smo[0], self._flurefs, self._partition),
                compute_flu(self._rings, self._molinfo, self._Smo[1], self._flurefs, self._partition)
            )
        return self._done_flus

    @property
    def flu(self):
        if self._flus()[0] is None:
            return None
        return self._flus()[0] + self._flus()[1]

    @property
    def flu_alpha(self):
        return self._flus()[0]

    @property
    def flu_beta(self):
        return self._flus()[1]

    def _boas(self):
        if not hasattr(self, '_done_boas'):
            self._done_boas = (
                compute_boa(self._rings, self._Smo[0]),
                compute_boa(self._rings, self._Smo[1])
            )
        return self._done_boas

    @property
    def boa(self):
        return self._boas()[0][0] + self._boas()[1][0]

    @property
    def boa_alpha(self):
        return self._boas()[0][0]

    @property
    def boa_beta(self):
        return self._boas()[1][0]

    @property
    def boa_c(self):
        return self._boas()[0][1] + self._boas()[1][1]

    @property
    def boa_c_alpha(self):
        return self._boas()[0][1]

    @property
    def boa_c_beta(self):
        return self._boas()[1][1]

    def _homas(self):
        if not hasattr(self, '_done_homas'):
            self._done_homas = compute_homa(self._rings, self._molinfo, self._homarefs)
        return self._done_homas

    @property
    def homa(self):
        if self._homas() is None:
            return None
        return self._homas()[0]

    @property
    def en(self):
        return self._homas()[1]

    @property
    def geo(self):
        return self._homas()[2]

    @property
    def homer(self):
        return compute_homer(self._rings, self._molinfo, self._homerrefs)

    def _blas(self):
        if not hasattr(self, '_done_blas'):
            self._done_blas = compute_bla(self._rings, self._molinfo)
        return self._done_blas

    @property
    def bla(self):
        return self._blas()[0]

    @property
    def bla_c(self):
        return self._blas()[1]

class IndicatorsNatorb:
    def __init__(self, Smo=None, rings=None, mol=None, mf=None, myhf=None, partition=None, mci=None, av1245=None, flurefs=None, homarefs=None, homerrefs=None, connectivity=None, geom=None, molinfo=None, ncores=1):
        self._Smo = Smo
        self._rings = rings
        self._mol = mol
        self._mf = mf
        self._myhf = myhf
        self._partition = partition
        self._mci = mci
        self._av1245 = av1245
        self._flurefs = flurefs
        self._homarefs = homarefs
        self._homerrefs = homerrefs
        self._connectivity = connectivity
        self._geom = geom
        self._molinfo = molinfo
        self._ncores = ncores

    @property
    def iring(self):
        return compute_iring_no(self._rings, self._Smo)

    @property
    def mci(self):
        if not hasattr(self, '_done_mci'):
            if self._ncores == 1:
                self._done_mci = sequential_mci_no(self._rings, self._Smo, self._partition)
            else:
                self._done_mci = multiprocessing_mci_no(self._rings, self._Smo, self._ncores, self._partition)
        return self._done_mci

    def _av_no(self):
        if not hasattr(self, '_done_av_no'):
            self._done_av_no = compute_av1245_no(self._rings, self._Smo, self._partition)
        return self._done_av_no

    @property
    def av1245(self):
        return self._av_no()[0]

    @property
    def avmin(self):
        return self._av_no()[1]

    @property
    def av1245_list(self):
        return self._av_no()[2]

    def _pdi_no(self):
        if not hasattr(self, '_done_pdi_no'):
            self._done_pdi_no = compute_pdi_no(self._rings, self._Smo)
        return self._done_pdi_no

    @property
    def pdi(self):
        return self._pdi_no()[0]

    @property
    def pdi_list(self):
        return self._pdi_no()[1]

    @property
    def flu(self):
        return compute_flu(self._rings, self._mol, self._Smo, self._flurefs, self._connectivity, self._partition)

    def _boa_no(self):
        if not hasattr(self, '_done_boa_no'):
            self._done_boa_no = compute_boa_no(self._rings, self._Smo)
        return self._done_boa_no

    @property
    def boa(self):
        return self._boa_no()[0]

    @property
    def boa_c(self):
        return self._boa_no()[1]

    @property
    def homer(self):
        if self._geom is None or self._homerrefs is None or self._connectivity is None:
            return None
        else:
            return compute_homer(self._rings, self._mol, self._geom, self._homerrefs, self._connectivity)

    def _homas(self):
        if not hasattr(self, '_done_homas'):
            self._done_homas = compute_homa(self._rings, self._molinfo, self._homarefs)
        return self._done_homas

    @property
    def homa(self):
        if self._homas() is None:
            return None
        return self._homas()[0]

    @property
    def en(self):
        return self._homas()[1]

    @property
    def geo(self):
        return self._homas()[2]

    def _blas(self):
        if not hasattr(self, '_done_blas'):
            self._done_blas = compute_bla(self._rings, self._molinfo)
        return self._done_blas

    @property
    def bla(self):
        return self._blas()[0]

    @property
    def bla_c(self):
        return self._blas()[1]

class ESI:
    """
    Main program for the ESIpy code.

    Attributes:
    Smo (concatenated list): Atomic Overlap Matrices (AOMs) in the MO basis.
    rings (list): List of indices of the atoms in the ring connectivity. Can be a list of lists.
    mol (optional, obj): Molecule object "mol" from PySCF.
    mf (optional, obj): Calculation object "mf" from PySCF.
    myhf (optional, obj): Reference object for Natural Orbitals calculation.
    partition (optional, str): Type of Hilbert-space partition scheme. Options are 'mulliken', 'lowdin', 'meta_lowdin', 'nao' and 'iao'.
    mci (optional, boolean): Whether to compute the MCI.
    av1245 (optional, boolean): Whether to compute the AV1245.
    flurefs (optional, dict): Custom FLU references.
    homarefs (optional, dict): Custom HOMA references: "n_opt", "c", "r1".
    homerrefs (optional, dict): Custom HOMER references: "alpha" and "r_opt" .
    connectivity (optional, list): Symbols of the ring connectivity as in "rings".
    geom (optional, list of lists): Geometry of the molecule as in mol.atom_coords().
    molinfo (optional, dict): Information about the molecule and calculation.
    ncores (optional, int): Number of cores to use for the MCI calculation. Default is 1.
    saveaoms (optional): Name where to save the AOMs in binary.
    savemolinfo (optional): Name where to save the molecular information dictionary in binary.
    name (optional, str): Name of the calculation. Default is 'calc'.
    readpath (optional, str): Path to read the AOMs. Default is '.'.

    indicators (obj): Object containing the indicators of the calculation.

    Methods:
    readaoms(): Reads the AOMs from a directory in AIMAll and ESIpy format.
    writeaoms(): Writes ESIpy's AOMs in AIMAll format.
    print(): Prints the output for the main ESIpy functions.
    """
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
        print("   Application to Aromaticity Calculations    ")
        print("  Joan Grebol, Eduard Matito, Pedro Salvador  ")
        print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")

        wf = wf_type(self.Smo)
        if isinstance(self.rings[0], int):
            self.rings = [self.rings]
        if wf == "rest":
            self.indicators = []
            for i in self.rings:
                self.indicators.append(IndicatorsRest(Smo=self.Smo, rings=i, mol=self.mol, mf=self.mf, myhf=self.myhf, partition=self.partition, mci=self.mci,
                     av1245=self.av1245, flurefs=self.flurefs, homarefs=self.homarefs, homerrefs=self.homerrefs, connectivity=self.connectivity, geom=self.geom,
                     molinfo=self.molinfo, ncores=self.ncores))
        elif wf == "unrest":
            self.indicators = []
            for i in self.rings:
                self.indicators.append(IndicatorsUnrest(Smo=self.Smo, rings=i, mol=self.mol, mf=self.mf, myhf=self.myhf, partition=self.partition, mci=self.mci,
                                             av1245=self.av1245, flurefs=self.flurefs, homarefs=self.homarefs, homerrefs=self.homerrefs, connectivity=self.connectivity, geom=self.geom,
                                             molinfo=self.molinfo, ncores=self.ncores))
        elif wf == "no":
            if np.ndim(self.mf.make_rdm1(ao_repr=True)) == 3:
                raise ValueError(" | Can not compute Natural Orbitals from unrestricted orbitals YET.")
            self.indicators = []
            for i in self.rings:
                self.indicators.append(IndicatorsNatorb(Smo=self.Smo, rings=i, mol=self.mol, mf=self.mf, myhf=self.myhf, partition=self.partition, mci=self.mci,
                                               av1245=self.av1245, flurefs=self.flurefs, homarefs=self.homarefs, homerrefs=self.homerrefs, connectivity=self.connectivity, geom=self.geom,
                                               molinfo=self.molinfo, ncores=self.ncores))
        else:
            raise ValueError(" | Could not determine the wavefunction type")

    @property
    def Smo(self):
        if isinstance(self._Smo, str):
            return load_file(self._Smo)
        if self.readpath != '.':
            return self.readaoms()
        if self._Smo is None:
            if isinstance(self.partition, list):
                raise ValueError(
                    " | Only one partition at a time. Partition should be a string, not a list.\n | Please consider looping through the partitions before calling the function")
            if self.mol and self.mf and self.partition:
                self._Smo = make_aoms(self.mol, self.mf, partition=self.partition, save=self.saveaoms, myhf=self.myhf)
                if self.saveaoms:
                    print(f" | Saved the AOMs in the {self.saveaoms} file")
            else:
                raise ValueError(" | Missing variables 'mol', 'mf', or 'partition'")
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

        Arguments:
            readpath (str): Path to the directory where the AOMs are stored. Default, '.'.

        Returns:
            The AOMs in the MO basis.
        """
        if self.name == "calc":
            print(" | No 'name' specified. Will use 'calc'")
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
            - For Natural Orbitals, a 'name.wfx' with the occupancies for the ESI-3D code.
        """
        for attr in ['mol', 'mf', 'name', 'Smo', 'partition']:
            if getattr(self, attr, None) is None:
                raise AttributeError(
                    f" | Missing required attribute '{attr}'. Please define it before calling ESI.writeaoms")

        write_aoms(self.mol, self.mf, self.name, self.Smo, self.rings, self.partition)
        shortpart = format_short_partition(self.partition)
        print(f" | Written the AOMs in {self.readpath}/{self.name}_{shortpart}/")

    def print(self):
        """
        Main output for ESIpy. Population analysis, localization and delocalization indices and
        electronic (and geometric) aromaticity indicators.
        """

        if self.molinfo is None or len(self.molinfo) == 1:
            if self.mol is None:
                raise ValueError(" | Missing 'mol' and/or 'molinfo'. Can not compute")
        if "symbols" not in self.molinfo:
            self.molinfo.update({"symbols": symbols})
        if "basisset" not in self.molinfo:
            if isinstance(self.molinfo["basisset"], dict):
                basisset = "Different basis sets"
            else:
                basisset = molinfo["basisset"]
            self.molinfo.update({"basisset": basisset})
        if "calctype" not in self.molinfo:
            self.molinfo.update({"calctype": "Not specified"})
        if "xc" not in self.molinfo:
            self.molinfo.update({"xc": "Not specified"})
        if "energy" not in self.molinfo:
            self.molinfo.update({"energy": "Not specified"})
        if "method" not in self.molinfo:
            self.molinfo.update({"method": "Not specified"})

        if isinstance(self.partition, list):
            raise ValueError(" | Only one partition at a time. Partition should be a string, not a list\n | Please consider looping through the partitions before calling the function")

        if self.rings is None:
            raise ValueError(" | The variable 'rings' is mandatory and must be a list with the ring connectivity")

        if self._Smo is None:
            if isinstance(self.Smo, str):
                print(f" | Loading the AOMs from file {self.Smo}")
                Smo = load_file(self.Smo)
                if Smo is None:
                    raise NameError(" | Please provide a valid name to read the AOMs")
            print(f" | Partition {self.partition} does not have Smo, generating it")
            self._Smo = self.Smo
            if self._Smo is None:
                raise ValueError(" | Could not build the AOMs from the given data")

        if wf_type(self.Smo) == "rest":
            from esipy.rest import info_rest, deloc_rest, arom_rest
            info_rest(self.Smo, self.molinfo)
            deloc_rest(self.Smo, self.molinfo)
            arom_rest(rings=self.rings, molinfo=self.molinfo, indicators=self.indicators, mci=self.mci, av1245=self.av1245,
                      flurefs=self.flurefs, homarefs=self.homarefs, homerrefs=self.homerrefs, ncores=self.ncores)

        elif wf_type(self.Smo) == "unrest":
            from esipy.unrest import info_unrest, deloc_unrest, arom_unrest
            info_unrest(self.Smo, self.molinfo)
            deloc_unrest(self.Smo, self.molinfo)
            arom_unrest(Smo=self.Smo, rings=self.rings, molinfo=self.molinfo, indicators=self.indicators, mci=self.mci, av1245=self.av1245, partition=self.partition,
                      flurefs=self.flurefs, homarefs=self.homarefs, homerrefs=self.homerrefs, ncores=self.ncores)

        elif wf_type(self.Smo) == "no":
            from esipy.no import info_no, deloc_no, arom_no
            info_no(self.Smo, self.molinfo)
            deloc_no(self.Smo, self.molinfo)
            arom_no(rings=self.rings, molinfo=self.molinfo, indicators=self.indicators, mci=self.mci, av1245=self.av1245,
                        flurefs=self.flurefs, homarefs=self.homarefs, homerrefs=self.homerrefs, ncores=self.ncores)


environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"
environ["MKL_NUM_THREADS"] = "1"

