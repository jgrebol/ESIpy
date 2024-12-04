from os import environ

from esipy.make_aoms import make_aoms
from esipy.atomicfiles import write_aoms, read_aoms
from esipy.tools import mol_info, format_partition, load_file, format_short_partition, wf_type
from esipy.indicators import *

class IndicatorsRest:
    def __init__(self, Smo=None, rings=None, mol=None, mf=None, myhf=None, partition=None, mci=None, av1245=None, flurefs=None, homarefs=None, homerrefs=None, connectivity=None, geom=None, molinfo=None, ncores=1, saveaoms=None, savemolinfo=None, name="calc", readpath='.'):
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
        self._saveaoms = saveaoms
        self._savemolinfo = savemolinfo
        self._name = name
        self._readpath = readpath

    @property
    def iring(self):
        return 2 * compute_iring(self._rings, self._Smo)

    @property
    def mci(self):
        if self._ncores == "1":
            return 2 * sequential_mci(self._rings, self._Smo, self._partition)
        else:
            return 2 * multiprocessing_mci(self._rings, self._Smo, self._ncores, self._partition)

    def _av1245(self):
        if not hasattr(self, '_done_av1245'):
            self._done_av1245 = esipy.indicators.compute_av1245(self._rings, self._Smo, self._partition)
        return self._done_av1245

    @property
    def av1245(self):
        return 2 * self._av1245()[0]

    @property
    def avmin(self):
        return 2 * self._av1245()[1]

    @property
    def av1245_list(self):
        return self._av1245()[2]

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
        if self._flurefs is None:
            print(" | Using default FLU references")
        else:
            print(" | Using FLU references provided by the user")
        return compute_flu(self._rings, self._molinfo, self._Smo, self._flurefs, self._partition)

    def _boa(self):
        if not hasattr(self, '_done_boa'):
            self._done_boa = esipy.indicators.compute_boa(self._rings, self._Smo)
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
            return esipy.indicators.compute_homer(self._rings, self._mol, self._geom, self._homerrefs, self._connectivity)

    def _homa(self):
        if not hasattr(self, '_done_homa'):
            self._done_homa = compute_homa(self._rings, self._molinfo, self._homarefs)
        return self._done_homa

    @property
    def homa(self):
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
    if self._geom is None or self._homerrefs is None or self._connectivity is None:
        return None
    else:
        return esipy.indicators.compute_homer(self._rings, self._mol, self._geom, self._homerrefs, self._connectivity)


class IndicatorsUnrest:
    def __init__(self, Smo=None, rings=None, mol=None, mf=None, myhf=None, partition=None, mci=None, av1245=None, flurefs=None, homarefs=None, homerrefs=None, connectivity=None, geom=None, molinfo=None, ncores=1, saveaoms=None, savemolinfo=None, name="calc", readpath='.'):
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
        self._saveaoms = saveaoms
        self._savemolinfo = savemolinfo
        self._name = name
        self._readpath = readpath

    def _irings(self):
        if not hasattr(self, '_done_irings'):
            self._done_irings = (
                esipy.indicators.compute_iring(self._rings, self._Smo[0]),
                esipy.indicators.compute_iring(self._rings, self._Smo[1])
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
                mci_alpha = esipy.indicators.sequential_mci(self._rings, self._Smo[0], self._partition)
                mci_beta = esipy.indicators.sequential_mci(self._rings, self._Smo[1], self._partition)
            else:
                mci_alpha = esipy.indicators.multiprocessing_mci(self._rings, self._Smo[0], self._ncores,
                                                                 self._partition)
                mci_beta = esipy.indicators.multiprocessing_mci(self._rings, self._Smo[1], self._ncores,
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

    def _av1245s(self):
        if not hasattr(self, '_done_av1245s'):
            self._done_av1245s = (
                esipy.indicators.compute_av1245(self._rings, self._Smo[0], self._partition),
                esipy.indicators.compute_av1245(self._rings, self._Smo[1], self._partition)
            )
        return self._done_av1245s

    @property
    def av1245(self):
        return self._av1245s()[0][0] + self._av1245s()[1][0]

    @property
    def av1245_alpha(self):
        return self._av1245s()[0][0]

    @property
    def av1245_beta(self):
        return self._av1245s()[1][0]

    @property
    def avmin(self):
        return self._av1245s()[0][1] + self._av1245s()[1][1]

    @property
    def avmin_alpha(self):
        return self._av1245s()[0][1]

    @property
    def avmin_beta(self):
        return self._av1245s()[1][1]

    @property
    def av1245_list(self):
        return self._av1245s()[0][2] + self._av1245s()[1][2]

    @property
    def av1245_list_alpha(self):
        return self._av1245s()[0][2]

    @property
    def av1245_list_beta(self):
        return self._av1245s()[1][2]

    def _pdis(self):
        if not hasattr(self, '_done_pdis'):
            self._done_pdis = (
                esipy.indicators.compute_pdi(self._rings, self._Smo[0]),
                esipy.indicators.compute_pdi(self._rings, self._Smo[1])
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

    @property
    def flu(self):
        return esipy.indicators.compute_flu(self._rings, self._mol, self._Smo[0], self._flurefs, self._connectivity, self._partition) + esipy.indicators.compute_flu(self._rings, self._mol, self._Smo[1], self._flurefs, self._connectivity, self._partition)

    @property
    def flu_alpha(self):
        return esipy.indicators.compute_flu(self._rings, self._mol, self._Smo[0], self._flurefs, self._connectivity, self._partition)

    @property
    def flu_beta(self):
        return esipy.indicators.compute_flu(self._rings, self._mol, self._Smo[1], self._flurefs, self._connectivity, self._partition)

    def _boas(self):
        if not hasattr(self, '_done_boas'):
            self._done_boas = (
                esipy.indicators.compute_boa(self._rings, self._Smo[0]),
                esipy.indicators.compute_boa(self._rings, self._Smo[1])
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

    @property
    def homer(self):
        if self._geom is None or self._homerrefs is None or self._connectivity is None:
            return None
        else:
            return esipy.indicators.compute_homer(self._rings, self._mol, self._geom, self._homerrefs, self._connectivity) + esipy.indicators.compute_homer(self._rings, self._mol, self._geom, self._homerrefs, self._connectivity)

    @property
    def homer_alpha(self):
        if self._geom is None or self._homerrefs is None or self._connectivity is None:
            return None
        else:
            return esipy.indicators.compute_homer(self._rings, self._mol, self._geom, self._homerrefs, self._connectivity)

    @property
    def homer_beta(self):
        if self._geom is None or self._homerrefs is None or self._connectivity is None:
            return None
        else:
            return esipy.indicators.compute_homer(self._rings, self._mol, self._geom, self._homerrefs, self._connectivity)


class IndicatorsNatorb:
    def __init__(self, Smo=None, rings=None, mol=None, mf=None, myhf=None, partition=None, mci=None, av1245=None, flurefs=None, homarefs=None, homerrefs=None, connectivity=None, geom=None, molinfo=None, ncores=1, saveaoms=None, savemolinfo=None, name="calc", readpath='.'):
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
        self._saveaoms = saveaoms
        self._savemolinfo = savemolinfo
        self._name = name
        self._readpath = readpath

    @property
    def iring(self):
        return esipy.indicators.compute_iring_no(self._rings, self._Smo)

    @property
    def mci(self):
        if self._ncores == "1":
            return esipy.indicators.sequential_mci_no(self._rings, self._Smo, self._partition)
        else:
            return esipy.indicators.multiprocessing_mci_no(self._rings, self._Smo, self._ncores, self._partition)

    def _av1245_no(self):
        if not hasattr(self, '_done_av1245_no'):
            self._done_av1245_no = esipy.indicators.compute_av1245_no(self._rings, self._Smo, self._partition)
        return self._done_av1245_no

    @property
    def av1245(self):
        return self._av1245_no()[0]

    @property
    def avmin(self):
        return self._av1245_no()[1]

    @property
    def av1245_list(self):
        return self._av1245_no()[2]

    def _pdi_no(self):
        if not hasattr(self, '_done_pdi_no'):
            self._done_pdi_no = esipy.indicators.compute_pdi_no(self._rings, self._Smo)
        return self._done_pdi_no

    @property
    def pdi(self):
        return self._pdi_no()[0]

    @property
    def pdi_list(self):
        return self._pdi_no()[1]

    @property
    def flu(self):
        return esipy.indicators.compute_flu(self._rings, self._mol, self._Smo, self._flurefs, self._connectivity, self._partition)

    def _boa_no(self):
        if not hasattr(self, '_done_boa_no'):
            self._done_boa_no = esipy.indicators.compute_boa_no(self._rings, self._Smo)
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
            return esipy.indicators.compute_homer(self._rings, self._mol, self._geom, self._homerrefs, self._connectivity)

    def _homas(self):
        if not hasattr(self, '_done_homas'):
            self._done_homas = esipy.indicators.compute_homa(self._rings, self._mol, self._geom, self._homarefs, self._connectivity)
        return self._done_homas

    @property
    def homa(self):
        return self._homas()[0]

    @property
    def en(self):
        return self._homas()[1]

    @property
    def geo(self):
        return self._homas()[2]

    def _blas(self):
        if not hasattr(self, '_done_blas'):
            self._done_blas = esipy.indicators.compute_bla(self._rings, self._mol, self._geom)
        return self._done_blas

    @property
    def bla(self):
        return self._blas()[0]

    @property
    def bla_c(self):
        return self._blas()[1]

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

        wf = wf_type(self.Smo)
        if wf == "rest":
            self.indicators = IndicatorsRest()

            self.indicators = IndicatorsRest(Smo=self.Smo, rings=self.rings, mol=self.mol, mf=self.mf, myhf=self.myhf, partition=self.partition, mci=self.mci,
                         av1245=self.av1245, flurefs=self.flurefs, homarefs=self.homarefs, homerrefs=self.homerrefs, connectivity=self.connectivity, geom=self.geom,
                         molinfo=self.molinfo, ncores=self.ncores, saveaoms=self.saveaoms, savemolinfo=self.savemolinfo, name=self.name, readpath=self.readpath)
        elif wf == "unrest":
            self.indicators = IndicatorsUnrest
        elif wf == "natorb":
            self.indicators = IndicatorsNatorb
        else:
            raise ValueError(" | Could not determine the wavefunction type")

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

    def print(self):
        """
        Main ESIpy. Calculation of population analysis, localization and delocalization indices and
        electronic (and geometric) aromaticity indicators.
        """
        partition = format_partition(self.partition)
        fromaoms = False

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

        print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")

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
            arom_rest(self.Smo, self.rings, self.molinfo, self.indicators, mci=self.mci, av1245=self.av1245, flurefs=self.flurefs, homarefs=self.homarefs,
                      homerrefs=self.homerrefs, connectivity=self.connectivity, geom=self.connectivity, ncores=self.ncores)
        elif wf_type(self.Smo) == "unrest":
            info_unrest()
            deloc_unrest()
            arom_unrest()
        elif wf_type(self.Smo) == "no":
            info_no()
            deloc_no()
            arom_no()


environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"
environ["MKL_NUM_THREADS"] = "1"


