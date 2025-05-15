from os import environ
from copy import deepcopy
import numpy as np
from time import time

from esipy.atomicfiles import write_aoms, read_aoms, read_molinfo
from esipy.indicators import (
    compute_iring, sequential_mci, multiprocessing_mci,
    compute_av1245, compute_pdi, compute_flu, compute_boa, compute_homer, compute_homa, compute_bla,
    compute_iring_no, sequential_mci_no, multiprocessing_mci_no, compute_av1245_no,
    compute_pdi_no, compute_boa_no
)
from esipy.make_aoms import make_aoms
from esipy.tools import (mol_info, format_partition, load_file, format_short_partition, wf_type, save_file,
                         build_connec_rest, build_connec_unrest, build_connec_no, find_rings, process_fragments, build_connectivity)


class IndicatorsRest:
    def __init__(self, aom=None, rings=None, mol=None, mf=None, myhf=None, partition=None, mci=None, av1245=None,
                 flurefs=None, homarefs=None, homerrefs=None, connectivity=None, geom=None, molinfo=None, ncores=1):
        """
        Initialize the indicators from Restricted calculations.

        Parameters:
            aom (concatenated list): Atomic Overlap Matrices (AOMs) in the MO basis.
            rings (list): List of indices of the atoms in the ring connectivity. Can be a list of lists.
            mol (optional, obj): Molecule object "mol" from PySCF.
            mf (optional, obj): Calculation object "mf" from PySCF.
            partition (optional, str): Type of Hilbert-space partition scheme.
                Options are 'mulliken', 'lowdin', 'meta_lowdin', 'nao' and 'iao'.
            mci (optional, boolean): Whether to compute the MCI.
            av1245 (optional, boolean): Whether to compute the AV1245.
            flurefs (optional, dict): Custom FLU references.
            homarefs (optional, dict): Custom HOMA references: "n_opt", "c", "r1".
            homerrefs (optional, dict): Custom HOMER references: "alpha" and "r_opt" .
            connectivity (optional, list): Symbols of the ring connectivity as in "rings".
            geom (optional, list of lists): Geometry of the molecule as in mol.atom_coords().
            molinfo (optional, dict): Information about the molecule and calculation.
            ncores (optional, int): Number of cores to use for the MCI calculation. Default is 1.
        """
        self._aom = aom
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
        """
        Compute the Iring value.

        :returns: The Iring value
            :rtype: float
        """
        return 2 * compute_iring(self._rings, self._aom)

    @property
    def mci(self):
        """
        Compute the MCI value.

        :returns: The MCI value
        :rtype: float
        """
        if not hasattr(self, '_done_mci'):
            if self._ncores == 1:
                self._done_mci = sequential_mci(self._rings, self._aom, self._partition)
            else:
                self._done_mci = multiprocessing_mci(self._rings, self._aom, self._ncores, self._partition)
        return 2 * self._done_mci

    def _av(self):
        """
        Compute the AV1245, AVmin and the list of the 4c-MCIs.

        :returns: Tuple containing the AV1245, AVmin and the list of the 4c-MCIs.
        :rtype: tuple
        """
        if not hasattr(self, '_done_av'):
            self._done_av = compute_av1245(self._rings, self._aom, self._partition)
        return self._done_av

    @property
    def av1245(self):
        """
        Get the AV1245 value.

        :returns: The AV1245 value.
        :rtype: float
        """
        return 2 * self._av()[0]

    @property
    def avmin(self):
        """
        Get the AVmin value.

        :returns: The AVmin value.
        :rtype: float
        """
        return 2 * self._av()[1]

    @property
    def av1245_list(self):
        """
        Get the list of 4c-MCIs that form the AV1245.

        :returns: The list of 4c-MCIs that form the AV1245.
        :rtype: numpy.ndarray
        """
        return 2 * np.array(self._av()[2], dtype=object)

    def _pdi(self):
        """
        Compute the PDI.

        :returns: The PDI value.
        :rtype: float
        """
        if not hasattr(self, '_done_pdi'):
            self._done_pdi = compute_pdi(self._rings, self._aom)
        return self._done_pdi

    @property
    def pdi(self):
        """
        Get the PDI value.

        :returns: The PDI value.
        :rtype: float
        """
        return 2 * self._pdi()[0]

    @property
    def pdi_list(self):
        """
        Get the list of the DIs (1-4, 2-5, 3-6).

        :returns: The list of the DI values that form PDI.
        :rtype: numpy.ndarray
        """
        return 2 * np.array(self._pdi()[1], dtype=object)

    @property
    def flu(self):
        """
        Compute the FLU value.

        :returns: The FLU value.
        :rtype: float
        """
        return compute_flu(self._rings, self._molinfo, self._aom, self._flurefs, self._partition)

    def _boa(self):
        """
        Compute the BOA and BOA_c values.

        :returns: The BOA and BOA_c values.
        :rtype: tuple
        """
        if not hasattr(self, '_done_boa'):
            self._done_boa = compute_boa(self._rings, self._aom)
        return self._done_boa

    @property
    def boa(self):
        """
        Get the BOA value.

        :returns: The BOA value.
        :rtype: float
        """
        return 2 * self._boa()[0]

    @property
    def boa_c(self):
        """
        Get the BOA_c value.

        :returns: The BOA_c value.
        :rtype: float
        """
        return 2 * self._boa()[1]

    @property
    def homer(self):
        """
        Compute the HOMER value.

        :returns: The HOMER value.
        :rtype: float
        """
        if self._geom is None or self._homerrefs is None or self._connectivity is None:
            return None
        else:
            return compute_homer(self._rings, self._molinfo, self._homerrefs)

    def _homa(self):
        """
        Compute the HOMA and the EN and GEO components.

        :returns: The HOMA, EN and GEO values.
        :rtype: tuple
        """
        if not hasattr(self, '_done_homa'):
            if self._molinfo["geom"] is None:
                return None
            self._done_homa = compute_homa(self._rings, self._molinfo, self._homarefs)
        return self._done_homa

    @property
    def homa(self):
        """
        Get the HOMA value.

        :returns: The HOMA value.
        :rtype: float
        """
        if self._homa() is None:
            return None
        return self._homa()[0]

    @property
    def en(self):
        """
        Get the EN value.

        :returns: The EN value.
        :rtype: float
        """
        return self._homa()[1]

    @property
    def geo(self):
        """
        Get the GEO value.

        :returns: The GEO value.
        :rtype: float
        """
        return self._homa()[2]

    def _bla(self):
        """
        Compute the BLA and BLA_c values.

        :returns: The BLA and BLA_c values.
        :rtype: tuple
        """
        if not hasattr(self, '_done_bla'):
            self._done_bla = compute_bla(self._rings, self._molinfo)
            if self._done_bla is None:
                return None
        return self._done_bla

    @property
    def bla(self):
        """
        Get the BLA value.

        :returns: The BLA value.
        :rtype: float
        """
        if self._bla() is None:
            return [None, None]
        return self._bla()[0]

    @property
    def bla_c(self):
        """
        Get the BLA_c value.

        :returns: The BLA_c value.
        :rtype: float
        """
        if self._bla() is None:
            return [None, None]
        return self._bla()[1]


class IndicatorsUnrest:
    """
    Initialize the indicators from Unrestricted calculations.

    Parameters:
        aom (concatenated list): Atomic Overlap Matrices (AOMs) in the MO basis.
        rings (list): List of indices of the atoms in the ring connectivity. Can be a list of lists.
        mol (optional, obj): Molecule object "mol" from PySCF.
        mf (optional, obj): Calculation object "mf" from PySCF.
        partition (optional, str): Type of Hilbert-space partition scheme.
            Options are 'mulliken', 'lowdin', 'meta_lowdin', 'nao' and 'iao'.
        mci (optional, boolean): Whether to compute the MCI.
        av1245 (optional, boolean): Whether to compute the AV1245.
        flurefs (optional, dict): Custom FLU references.
        homarefs (optional, dict): Custom HOMA references: "n_opt", "c", "r1".
        homerrefs (optional, dict): Custom HOMER references: "alpha" and "r_opt" .
        connectivity (optional, list): Symbols of the ring connectivity as in "rings".
        geom (optional, list of lists): Geometry of the molecule as in mol.atom_coords().
        molinfo (optional, dict): Information about the molecule and calculation.
        ncores (optional, int): Number of cores to use for the MCI calculation. Default is 1.
    """

    def __init__(self, aom=None, rings=None, mol=None, mf=None, myhf=None, partition=None, mci=None, av1245=None,
                 flurefs=None, homarefs=None, homerrefs=None, connectivity=None, geom=None, molinfo=None, ncores=1):
        self._aom = aom
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
        """
        Compute the Iring and their alpha and beta components.

        :returns: The Iring, Iring_alpha and Iring_beta.
        :rtype: tuple
        """
        if not hasattr(self, '_done_irings'):
            self._done_irings = (
                compute_iring(self._rings, self._aom[0]),
                compute_iring(self._rings, self._aom[1])
            )
        return self._done_irings

    @property
    def iring(self):
        """
        Get the Iring value.

        :returns: The Iring value.
        :rtype: float
        """
        return self._irings()[0] + self._irings()[1]

    @property
    def iring_alpha(self):
        """
        Get the Iring_alpha value.

        :returns: The Iring_alpha value.
        :rtype: float
        """
        return self._irings()[0]

    @property
    def iring_beta(self):
        """
        Get the Iring_beta value.

        :returns: The Iring_beta value.
        :rtype: float
        """
        return self._irings()[1]

    def _mcis(self):
        """
        Compute the MCI values for alpha and beta components. Different algorithms are used depending on
            the number of cores.

        :returns: The MCI values for alpha and beta components.
        :rtype: tuple
        """
        if not hasattr(self, '_done_mcis'):
            if self._ncores == 1:
                mci_alpha = sequential_mci(self._rings, self._aom[0], self._partition)
                mci_beta = sequential_mci(self._rings, self._aom[1], self._partition)
            else:
                mci_alpha = multiprocessing_mci(self._rings, self._aom[0], self._ncores,
                                                self._partition)
                mci_beta = multiprocessing_mci(self._rings, self._aom[1], self._ncores,
                                               self._partition)
            self._done_mcis = (mci_alpha, mci_beta)
        return self._done_mcis

    @property
    def mci(self):
        """
        Get the MCI value.

        :returns: The MCI value.
        :rtype: float
        """
        return self._mcis()[0] + self._mcis()[1]

    @property
    def mci_alpha(self):
        """
        Get the MCI_alpha value.

        :returns: The MCI_alpha value.
        :rtype: float
        """
        return self._mcis()[0]

    @property
    def mci_beta(self):
        """
        Get the MCI_beta value.

        :returns: The MCI_beta value.
        :rtype: float
        """
        return self._mcis()[1]

    def _avs(self):
        """
        Compute the AV1245, AVmin and the list of the 4c-MCIs for alpha and beta components.

        :returns: The AV1245, AVmin and the list of the 4c-MCIs for alpha and beta components.
        :rtype: tuple
        """
        if not hasattr(self, '_done_avs'):
            self._done_avs = (
                compute_av1245(self._rings, self._aom[0], self._partition),
                compute_av1245(self._rings, self._aom[1], self._partition)
            )
        return self._done_avs

    @property
    def av1245(self):
        """
        Get the AV1245 value.

        :returns: The AV1245 value.
        :rtype: float
        """
        return self._avs()[0][0] + self._avs()[1][0]

    @property
    def av1245_alpha(self):
        """
        Get the AV1245_alpha value.

        :returns: The AV1245_alpha value.
        :rtype: float
        """
        return self._avs()[0][0]

    @property
    def av1245_beta(self):
        """
        Get the AV1245_beta value.

        :returns: The AV1245_beta value.
        :rtype: float
        """
        return self._avs()[1][0]

    @property
    def avmin(self):
        """
        Get the AVmin value.

        :returns: The AVmin value.
        :rtype: float
        """
        return min(list(self.av1245_list), key=abs)

    @property
    def avmin_alpha(self):
        """
        Get the AVmin_alpha value.

        :returns: The AVmin_alpha value.
        :rtype: float
        """
        return min(self.av1245_list_alpha, key=abs)

    @property
    def avmin_beta(self):
        """
        Get the AVmin_beta value.

        :returns: The AVmin_beta value.
        :rtype: float
        """
        return min(self.av1245_list_beta, key=abs)

    @property
    def av1245_list(self):
        """
        Get the list of 4c-MCIs that form the AV1245.

        :returns: The list of 4c-MCIs that form the AV1245.
        :rtype: numpy.ndarray
        """
        return np.add(self.av1245_list_alpha, self.av1245_list_beta)

    @property
    def av1245_list_alpha(self):
        """
        Get the list of 4c-MCIs that form the AV1245_alpha.

        :returns: The list of 4c-MCIs that form the AV1245_alpha.
        :rtype: numpy.ndarray
        """
        return self._avs()[0][2]

    @property
    def av1245_list_beta(self):
        """
        Get the list of 4c-MCIs that form the AV1245_beta.

        :returns: The list of 4c-MCIs that form the AV1245_beta.
        :rtype: numpy.ndarray
        """
        return self._avs()[1][2]

    def _pdis(self):
        """
        Compute the PDI values for alpha and beta components.

        :returns: The PDI values for alpha and beta components.
        :rtype: tuple
        """
        if not hasattr(self, '_done_pdis'):
            self._done_pdis = (
                compute_pdi(self._rings, self._aom[0]),
                compute_pdi(self._rings, self._aom[1])
            )
        return self._done_pdis

    @property
    def pdi(self):
        """
        Get the PDI value.

        :returns: The PDI value.
        :rtype: float
        """
        return self._pdis()[0][0] + self._pdis()[1][0]

    @property
    def pdi_alpha(self):
        """
        Get the PDI_alpha value.

        :returns: The PDI_alpha value.
        :rtype: float
        """
        return self._pdis()[0][0]

    @property
    def pdi_beta(self):
        """
        Get the PDI_beta value.

        :returns: The PDI_beta value.
        :rtype: float
        """
        return self._pdis()[1][0]

    @property
    def pdi_list(self):
        """
        Get the list of the DIs (1-4, 2-5, 3-6).

        :returns: The list of the DI values that form PDI.
        :rtype: numpy.ndarray
        """
        return self._pdis()[0][1] + self._pdis()[1][1]

    @property
    def pdi_list_alpha(self):
        """
        Get the list of the alpha component of the DIs (1-4, 2-5, 3-6).

        :returns: The list of the alpha component of the DI values that form PDI.
        :rtype: numpy.ndarray
        """
        return self._pdis()[0][1]

    @property
    def pdi_list_beta(self):
        """
        Get the list of the beta component of the DIs (1-4, 2-5, 3-6).

        :returns: The list of the beta component of the DI values that form PDI.
        :rtype: numpy.ndarray
        """
        return self._pdis()[1][1]

    def _flus(self):
        """
        Compute the FLU values for alpha and beta components.

        :returns: The FLU values for alpha and beta components.
        :rtype: tuple
        """
        if not hasattr(self, '_done_flus'):
            self._done_flus = (
                compute_flu(self._rings, self._molinfo, self._aom[0], self._flurefs, self._partition),
                compute_flu(self._rings, self._molinfo, self._aom[1], self._flurefs, self._partition)
            )
        return self._done_flus

    @property
    def flu(self):
        """
        Get the FLU value.

        :returns: The FLU value.
        :rtype: float
        """
        if self._flus()[0] is None:
            return None
        return self._flus()[0] + self._flus()[1]

    @property
    def flu_alpha(self):
        """
        Get the FLU_alpha value.

        :returns: The FLU_alpha value.
        :rtype: float
        """
        return self._flus()[0]

    @property
    def flu_beta(self):
        """
        Get the FLU_beta value.

        :returns: The FLU_beta value.
        :rtype: float
        """
        return self._flus()[1]

    def _boas(self):
        """
        Compute the BOA and BOA_c values for alpha and beta components.

        :returns: The BOA and BOA_c values for alpha and beta components.
        :rtype: tuple
        """
        if not hasattr(self, '_done_boas'):
            self._done_boas = (
                compute_boa(self._rings, self._aom[0]),
                compute_boa(self._rings, self._aom[1])
            )
        return self._done_boas

    @property
    def boa(self):
        """
        Get the BOA value.

        :returns: The BOA value.
        :rtype: float
        """
        return self._boas()[0][0] + self._boas()[1][0]

    @property
    def boa_alpha(self):
        """
        Get the BOA_alpha value.

        :returns: The BOA_alpha value.
        :rtype: float
        """
        return self._boas()[0][0]

    @property
    def boa_beta(self):
        """
        Get the BOA_beta value.

        :returns: The BOA_beta value
        :rtype: float
        """
        return self._boas()[1][0]

    @property
    def boa_c(self):
        """
        Get the BOA_c value.

        :returns: The BOA_c value.
        :rtype: float
        """
        return self._boas()[0][1] + self._boas()[1][1]

    @property
    def boa_c_alpha(self):
        """
        Get the BOA_c_alpha value.

        :returns: The BOA_c_alpha value.
        :rtype: float
        """
        return self._boas()[0][1]

    @property
    def boa_c_beta(self):
        """
        Get the BOA_c_beta value.

        :returns: The BOA_c_beta value.
        :rtype: float
        """
        return self._boas()[1][1]

    def _homas(self):
        """
        Compute the HOMA and the EN and GEO components.

        :returns: The HOMA, EN and GEO values.
        :rtype: tuple
        """
        if not hasattr(self, '_done_homas'):
            self._done_homas = compute_homa(self._rings, self._molinfo, self._homarefs)
        return self._done_homas

    @property
    def homa(self):
        """
        Get the HOMA value.

        :returns: The HOMA value.
        :rtype: float
        """
        if self._homas() is None:
            return None
        return self._homas()[0]

    @property
    def en(self):
        """
        Get the EN value.

        :returns: The EN value.
        :rtype: float
        """
        return self._homas()[1]

    @property
    def geo(self):
        """
        Get the GEO value.

        :returns: The GEO value.
        :rtype: float
        """
        return self._homas()[2]

    @property
    def homer(self):
        """
        Get the HOMER value.

        :returns: The HOMER value.
        :rtype: float
        """
        return compute_homer(self._rings, self._molinfo, self._homerrefs)

    def _blas(self):
        """
        Compute the BLA and BLA_c values.

        :returns: The BLA and BLA_c values.
        :rtype: tuple
        """
        if not hasattr(self, '_done_blas'):
            self._done_blas = compute_bla(self._rings, self._molinfo)
        return self._done_blas

    @property
    def bla(self):
        """
        Get the BLA value.

        :returns: The BLA value.
        :rtype: float
        """
        return self._blas()[0]

    @property
    def bla_c(self):
        """
        Get the BLA_c value.

        :returns: The BLA_c value.
        :rtype: float
        """
        return self._blas()[1]


class IndicatorsNatorb:
    """
    Initialize the indicators from Natural Orbitals calculations.

    Parameters:
        aom (concatenated list): Atomic Overlap Matrices (AOMs) in the MO basis.
        rings (list): List of indices of the atoms in the ring connectivity. Can be a list of lists.
        mol (optional, obj): Molecule object "mol" from PySCF.
        mf (optional, obj): Calculation object "mf" from PySCF.
        myhf (optional, obj): Reference RHF object for IAO-Natural Orbitals calculation.
        partition (optional, str): Type of Hilbert-space partition scheme.
            Options are 'mulliken', 'lowdin', 'meta_lowdin', 'nao' and 'iao'.
        mci (optional, boolean): Whether to compute the MCI.
        av1245 (optional, boolean): Whether to compute the AV1245.
        flurefs (optional, dict): Custom FLU references.
        homarefs (optional, dict): Custom HOMA references: "n_opt", "c", "r1".
        homerrefs (optional, dict): Custom HOMER references: "alpha" and "r_opt" .
        connectivity (optional, list): Symbols of the ring connectivity as in "rings".
        geom (optional, list of lists): Geometry of the molecule as in mol.atom_coords().
        molinfo (optional, dict): Information about the molecule and calculation.
        ncores (optional, int): Number of cores to use for the MCI calculation. Default is 1.
    """

    def __init__(self, aom=None, rings=None, mol=None, mf=None, myhf=None, partition=None, mci=None, av1245=None,
                 flurefs=None, homarefs=None, homerrefs=None, connectivity=None, geom=None, molinfo=None, ncores=1):
        self._aom = aom
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
        """
        Compute the Iring value.

        :returns: The Iring value.
        :rtype: float
        """
        return compute_iring_no(self._rings, self._aom)

    @property
    def mci(self):
        """
        Compute the MCI value.

        :returns: The MCI value.
        :rtype: float
        """
        if not hasattr(self, '_done_mci'):
            if self._ncores == 1:
                self._done_mci = sequential_mci_no(self._rings, self._aom, self._partition)
            else:
                self._done_mci = multiprocessing_mci_no(self._rings, self._aom, self._ncores, self._partition)
        return self._done_mci

    def _av_no(self):
        """
        Compute the AV1245, AVmin and the list of the 4c-MCIs.

        :returns: List containing the AV1245, AVmin and the list of the 4c-MCIs.
        :rtype: list
        """
        if not hasattr(self, '_done_av_no'):
            self._done_av_no = compute_av1245_no(self._rings, self._aom, self._partition)
        return self._done_av_no

    @property
    def av1245(self):
        """
        Get the AV1245 value.

        :returns: The AV1245 value
        :rtype: float
        """
        return self._av_no()[0]

    @property
    def avmin(self):
        """
        Get the AVmin value.

        :returns: The AVmin value.
        :rtype: float
        """
        return self._av_no()[1]

    @property
    def av1245_list(self):
        """
        Get the list of 4c-MCIs that form the AV1245.

        :returns: The list of 4c-MCIs that form the AV1245.
        :rtype: numpy.ndarray
        """
        return self._av_no()[2]

    def _pdi_no(self):
        """
        Compute the PDI.

        :returns: The PDI value.
        :rtype: float
        """
        if not hasattr(self, '_done_pdi_no'):
            self._done_pdi_no = compute_pdi_no(self._rings, self._aom)
        return self._done_pdi_no

    @property
    def pdi(self):
        """
        Get the PDI value.

        :returns: The PDI value.
        :rtype: float
        """
        return self._pdi_no()[0]

    @property
    def pdi_list(self):
        """
        Get the list of the DIs (1-4, 2-5, 3-6).

        :returns: The list of the DI values that form PDI.
        :rtype: numpy.ndarray
        """
        return self._pdi_no()[1]

    @property
    def flu(self):
        """
        Compute the FLU value.

        :returns: The FLU value.
        :rtype: float
        """
        return compute_flu(self._rings, self._mol, self._aom, self._flurefs, self._connectivity, self._partition)

    def _boa_no(self):
        """
        Compute the BOA and BOA_c values.

        :returns: The BOA and BOA_c values.
        :rtype: tuple
        """
        if not hasattr(self, '_done_boa_no'):
            self._done_boa_no = compute_boa_no(self._rings, self._aom)
        return self._done_boa_no

    @property
    def boa(self):
        """
        Get the BOA value.

        :returns: The BOA value.
        :rtype: float
        """
        return self._boa_no()[0]

    @property
    def boa_c(self):
        """
        Get the BOA_c value.

        :returns: The BOA_c value.
        :rtype: float
        """
        return self._boa_no()[1]

    @property
    def homer(self):
        """
        Compute the HOMER value.

        :returns: The HOMER value.
        :rtype: float
        """
        if self._geom is None or self._homerrefs is None or self._connectivity is None:
            return None
        else:
            return compute_homer(self._rings, self._mol, self._geom, self._homerrefs, self._connectivity)

    def _homas(self):
        """
        Compute the HOMA and the EN and GEO components.

        :returns: The HOMA, EN and GEO values.
        :rtype: tuple
        """
        if not hasattr(self, '_done_homas'):
            self._done_homas = compute_homa(self._rings, self._molinfo, self._homarefs)
        return self._done_homas

    @property
    def homa(self):
        """
        Get the HOMA value.

        :returns: The HOMA value.
        :rtype: float
        """
        if self._homas() is None:
            return None
        return self._homas()[0]

    @property
    def en(self):
        """
        Get the EN value.

        :returns: The EN value.
        :rtype: float
        """
        return self._homas()[1]

    @property
    def geo(self):
        """
        Get the GEO value.

        :returns: The GEO value.
        :rtype: float
        """
        return self._homas()[2]

    def _blas(self):
        """
        Compute the BLA and BLA_c values.

        :returns: The BLA and BLA_c values.
        :rtype: tuple
        """
        if not hasattr(self, '_done_blas'):
            self._done_blas = compute_bla(self._rings, self._molinfo)
        return self._done_blas

    @property
    def bla(self):
        """
        Get the BLA value.

        :returns: The BLA value.
        :rtype: float
        """
        return self._blas()[0]

    @property
    def bla_c(self):
        """
        Get the BLA_c value.

        :returns: The BLA_c value.
        :rtype: float
        """
        return self._blas()[1]


class ESI:
    """
    Main class for the ESIpy code.

    Attributes:
    aom (concatenated list): Atomic Overlap Matrices (AOMs) in the MO basis.
    rings (list): List of indices of the atoms in the ring connectivity. Can be a list of lists.
    mol (optional, obj): Molecule object "mol" from PySCF.
    mf (optional, obj): Calculation object "mf" from PySCF.
    myhf (optional, obj): Reference RHF object for IAO-Natural Orbitals calculation.
    partition (optional, str): Type of Hilbert-space partition scheme.
        Options are 'mulliken', 'lowdin', 'meta_lowdin', 'nao' and 'iao'.
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
    indicators (obj): Object containing the indicators of the calculation. Generated in the initialization.

    Methods:
    readaoms(): Reads the AOMs from a directory in AIMAll and ESIpy format.
    writeaoms(): Writes ESIpy's AOMs in AIMAll format.
    print(): Prints the output for the main ESIpy functions.
    """

    def __init__(self, aom=None, rings=None, mol=None, mf=None, myhf=None, partition=None,
                 mci=None, av1245=None, flurefs=None, homarefs=None,
                 homerrefs=None, connectivity=None, geom=None, molinfo={},
                 ncores=1, save=None, readpath='.', read=False,
                 maxlen=12, minlen=6, rings_thres=0.3,
                 ):
        # For usual ESIpy calculations
        self._aom = aom
        self._aom_loaded = False
        self._rings = rings
        self.mol = mol
        self.mf = mf
        self.myhf = myhf
        self._molinfo = molinfo
        self._partition = partition
        self._mci = mci
        self._av1245 = av1245
        # For custom references
        self.flurefs = flurefs
        self.homarefs = homarefs
        self.homerrefs = homerrefs
        self.connectivity = connectivity
        self.geom = geom
        self.frag = False
        # For other tools
        self.ncores = ncores
        self.save = save
        self.saveaoms = save + '_' + self.partition + ".aoms" if save else None
        self.savemolinfo = save + '_' + self.partition + ".molinfo" if save else None
        self.readpath = readpath
        self.read = read
        # For finding rings
        self.fragmap = {}
        self.rings_thres = rings_thres
        self.maxlen = maxlen
        self.minlen = minlen
        self._printedrings = False
        self._connec = None
        self.done_connec = False
        self.filtrings = []

        print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")
        print(" ** Localization & Delocalization Indices **  ")
        print(" ** For Hilbert-Space Atomic Partitioning **  ")
        print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")
        print("   Application to Aromaticity Calculations    ")
        print("  Joan Grebol, Eduard Matito, Pedro Salvador  ")
        print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")

        # Can not work on IAO and Natural Orbitals yet
        sd = ["RHF", "SymAdaptedRHF", "UHF", "SymAdaptedUHF", "RKS", "SymAdaptedRKS", "UKS", "SymAdaptedUKS"]
        if self.partition == "iao" and self.molinfo["calctype"] not in sd:
            print(" | WARNING: IAO and Natural Orbitals from unrestricted orbitals not implemented yet")
        elif self.partition == "nao" and self.molinfo["calctype"] not in sd:
            print(" | WARNING: NAO and Natural Orbitals from unrestricted orbitals not implemented yet")


        wf = wf_type(self.aom)
        if wf == "rest":
            if not self.rings:
                return
            self.fragaom, self.fragmap = list(process_fragments(self.aom, self.rings, False))
            if self.fragaom:
                self.totalaom = self.aom + self.fragaom
                self.nfrags = len(self.fragaom)
            else:
                self.totalaom = self.aom
                self.nfrags = 0

            self.indicators = []
            for i in deepcopy(self.rings):
                ring = []
                for j in range(len(i)):
                    ring.append(self.fragmap[tuple(i[j % len(i)])] if isinstance(i[j % len(i)], set) else i[j % len(i)])
                self.indicators.append(IndicatorsRest(aom=self.totalaom, rings=ring, mol=self.mol, mf=self.mf, myhf=self.myhf,
                                                      partition=self.partition, mci=self.mci,
                                                      av1245=self.av1245, flurefs=self.flurefs, homarefs=self.homarefs,
                                                      homerrefs=self.homerrefs, connectivity=self.connectivity,
                                                      geom=self.geom,
                                                      molinfo=self.molinfo, ncores=self.ncores))
        elif wf == "unrest":
            if not self.rings:
                return

            self.fragaom_a, self.fragmap = list(process_fragments(self.aom[0], self.rings, False))
            self.fragaom_b, self.fragmap = list(process_fragments(self.aom[1], self.rings, True))
            if self.fragaom_a:
                self.totalaom_a = self.aom[0] + self.fragaom_a
                self.totalaom_b = self.aom[1] + self.fragaom_b
                self.fragaom = [self.fragaom_a, self.fragaom_b]
                self.totalaom = [self.totalaom_a, self.totalaom_b]
                self.nfrags = len(self.fragaom_a)

            else:
                self.totalaom = self.aom
                self.nfrags = 0

            self.indicators = []
            for i in deepcopy(self.rings):
                ring = []
                for j in range(len(i)):
                    ring.append(self.fragmap[tuple(i[j % len(i)])] if isinstance(i[j % len(i)], set) else i[j % len(i)])
                self.indicators.append(IndicatorsUnrest(aom=self.totalaom, rings=ring, mol=self.mol, mf=self.mf, myhf=self.myhf,
                                                        partition=self.partition, mci=self.mci,
                                                        av1245=self.av1245, flurefs=self.flurefs,
                                                        homarefs=self.homarefs, homerrefs=self.homerrefs,
                                                        connectivity=self.connectivity, geom=self.geom,
                                                        molinfo=self.molinfo, ncores=self.ncores))
        elif wf == "no":
            if not self.rings:
                return
            if self.read is not True and type(self._aom) != str:
                if self.mf is None:
                    raise ValueError(" | Missing variable 'mf'.")
                if np.ndim(self.mf.make_rdm1(ao_repr=True)) == 3:
                    if self.partition == "nao":
                        raise ValueError(" | Can not compute Natural Orbitals and NAO from unrestricted orbitals YET.")
                    elif self.partition == "iao":
                        raise ValueError(" | Can not compute Natural Orbitals and IAO from unrestricted orbitals YET.")
            aom, occ = self.aom
            self.fragaom, self.fragmap = list(process_fragments(aom, self.rings, False))
            if self.fragaom:
                self.totalaom = aom + self.fragaom
                self.nfrags = len(self.fragaom)
            else:
                self.totalaom = aom
                self.nfrags = 0
            self.totalaom = [self.totalaom, occ]

            self.indicators = []
            for i in deepcopy(self.rings):
                ring = []
                for j in range(len(i)):
                    ring.append(self.fragmap[tuple(i[j % len(i)])] if isinstance(i[j % len(i)], set) else i[j % len(i)])
                self.indicators.append(IndicatorsNatorb(aom=self.totalaom, rings=ring, mol=self.mol, mf=self.mf, myhf=self.myhf,
                                                        partition=self.partition, mci=self.mci,
                                                        av1245=self.av1245, flurefs=self.flurefs,
                                                        homarefs=self.homarefs, homerrefs=self.homerrefs,
                                                        connectivity=self.connectivity, geom=self.geom,
                                                        molinfo=self.molinfo, ncores=self.ncores))
        else:
            raise ValueError(" | Could not determine the wavefunction type")

    @property
    def rings(self):
        if not self._rings:
            self.totalaom = self.aom
            self.nfrags = 0
            self.fragmap = {}
            return None
        if self._rings == "find" or self._rings == "f":
            if not self._printedrings:
                print(" | Finding rings...")
            startrings = time()
            if self.connec:
                graph = self.connec
            else:
                graph = self.molinfo.get("connec")
            if not graph:
                raise ValueError(" | Could not find the connectivity matrix. ")

            self._rings = find_rings(graph, self.minlen, self.maxlen)
            endrings = time()

            if not self._rings:
                raise ValueError(" | Could not find any ring. Please check the minimum and maximum ring lengths.")
            elif not self._printedrings:
                print(f" | Found {len(self._rings)} rings in {endrings-startrings} seconds:")
                print(" | -------------------------------")
                print(" | rings = [")
                for i in self._rings:
                    print(" | ", i, ",")
                print(" | ]")
                print(" | -------------------------------")
                self._printedrings = True
        if isinstance(self._rings[0], (int, set)):
            self._rings = [self._rings]

        if self.fragmap:
            self.filtrings = []
            for i in range(len(self._rings)):
                ring = []
                for j in range(len(self._rings[i])):
                    ring.append(self.fragmap[tuple(self._rings[i][j % len(self._rings[i])])] if isinstance(self._rings[i][j % len(self._rings[i])], set) else self._rings[i][j % len(self._rings[i])])
                self.filtrings.append(ring)
        if not self.filtrings:
            self.filtrings = self._rings
        return self._rings

    @rings.setter
    def rings(self, value):
        self._rings = value

    @property
    def molinfo(self):
        """
        Get the information about the molecule and calculation. If not provided, it will compute it. If set as a string,
        it will read the information from the directory. Can be saved in a file with the `savemolinfo` attribute.

        :returns: Information about the molecule and calculation.
        :rtype: dict
        """

        if self.read is True:
            self._molinfo = read_molinfo(self.readpath)
            return self._molinfo
        if isinstance(self._molinfo, str):
            self._molinfo = load_file(self._molinfo)
            return self._molinfo
        if not self._molinfo:
            if self.partition in ['mulliken', 'lowdin']:
                print(" | Building meta-Lowdin AOMs to compute connectivity.")
                if self.mol is None or self.mf is None:
                    raise ValueError(" | Missing variables 'mol' and 'mf'. Could not build meta-Lowdin AOMs.")
                mat = make_aoms(self.mol, self.mf, partition="meta_lowdin", save=None, myhf=self.myhf)
            else:
                mat = self.aom
            graph = build_connectivity(mat=mat, threshold=self.rings_thres)
            self._molinfo = mol_info(self.mol, self.mf, self.savemolinfo, self._partition, graph)

        return self._molinfo

    @property
    def aom(self):
        """
        Get the Atomic Overlap Matrices (AOMs) in the MO basis. If not provided, it will compute them. If set as
        a string, it will read the AOMs from the directory. Can be saved in a file with the `saveaoms` attribute.

        :returns: The AOMs in the MO basis.
        :rtype: list
        """

        if self._aom_loaded:
            return self._aom
        if isinstance(self._aom, str):
            return load_file(self._aom)
        if self.read == True:
            self._aom_loaded = True
            aom = self.readaoms()
            if self.save:
                import os
                save_file(aom, os.path.join(self.readpath, self.saveaoms))
                save_file(read_molinfo(self.readpath), os.path.join(self.readpath, self.savemolinfo))
                print(f" | Saved the AOMs in the {self.saveaoms} file")
                print(f" | Saved the molinfo in the {self.savemolinfo} file")
            return aom
        if self._aom is None:
            if isinstance(self.partition, list):
                raise ValueError(
                    " | Only one partition at a time. Partition should be a string, not a list.\n | Please consider looping through the partitions before calling the function")
            if self.mol and self.mf and self.partition:
                self._aom_loaded = True
                self._aom = make_aoms(self.mol, self.mf, partition=self.partition, save=self.saveaoms, myhf=self.myhf)
                if self.saveaoms:
                    print(f" | Saved the AOMs in the {self.saveaoms} file")
            else:
                raise ValueError(" | Missing variables 'mol', 'mf', or 'partition'")
        return self._aom

    @property
    def partition(self):
        """
        Get the partition scheme for the Hilbert-space. Options are 'mulliken', 'lowdin', 'meta_lowdin', 'nao' and 'iao'.
        Some variations of these names are also available.

        :returns: The partition scheme for the Hilbert-space calculation.
        :rtype: str
        """

        if isinstance(self._partition, str):
            return format_partition(self._partition)
        raise ValueError(" | Partition could not be processed. Options are 'mulliken', 'lowdin', 'meta_lowdin', 'nao' and 'iao'")

    @property
    def connec(self):
        """
        Get the connectivity matrix. If the partition is 'mulliken' or 'lowdin',
        build the meta-Lowdin AOMs and compute the connectivity matrix from there.
        The computation is controlled by the 'done_connec' flag.

        :returns: The connectivity matrix.
        :rtype: dict
        """
        if self._connec is not None:
            return self._connec
        if not hasattr(self, 'done_connec') or not self.done_connec:
            if self.molinfo.get("connec") is not None:
                self._connec = self.molinfo.get("connec")
                self.done_connec = True
                return self._connec

            if self.partition in ['mulliken', 'lowdin']:
                print(" | Building meta-Lowdin AOMs to compute connectivity.")
                if self.mol is None or self.mf is None:
                    raise ValueError(" | Missing variables 'mol' and 'mf'. Could not build meta-Lowdin AOMs.")
                mat = make_aoms(self.mol, self.mf, partition="meta_lowdin", save=None, myhf=self.myhf)
            else:
                mat = self.aom
            self._connec = build_connectivity(mat=mat, threshold=self.rings_thres)
            self.done_connec = True
            return self._connec

    @property
    def mci(self):
        """
        Whether to compute the MCI. If not provided, it will compute it if the number of rings is less than 12.

        :returns: Whether to compute the MCI.
        :rtype: boolean
        """
        if self._mci is not None:
            return self._mci
        if isinstance(self.rings[0], list):  # Check if there are more than one rings
            maxring = max(len(ring) for ring in self.rings)
        else:
            maxring = len(self.rings)
        return maxring < 12

    @mci.setter
    def mci(self, value):
        self._mci = value

    @property
    def av1245(self):
        """
        Whether to compute the AV1245. If not provided, it will compute it if the number of rings is greater than 9.

        :returns: Whether to compute the AV1245.
        :rtype: boolean
        """
        if self._av1245 is not None:
            return self._av1245
        if isinstance(self.rings[0], list):  # Check if there are more than one rings
            maxring = max(len(ring) for ring in self.rings)
        else:
            maxring = len(self.rings)
        return maxring > 9

    @av1245.setter
    def av1245(self, value):
        self._av1245 = value

    def readaoms(self):
        """
        Reads the AOMs from a directory in AIMAll and ESIpy format. Overwrites 'ESI.aom' variable. By default, it will
        read the AOMs from the working directory. If the 'readpath' attribute is set, it will read from that directory.

        :returns: The AOMs in the MO basis, overwriting the aom variable.
        :rtype: list
        """

        if self.readpath is None:
            print(" | No path specified in 'ESI.readpath'. Will assume working directory")

        self._aom = read_aoms(path=self.readpath)
        self._molinfo = read_molinfo(path=self.readpath)
        print(f" | Read the AOMs from {self.readpath}/")
        return self._aom

    def writeaoms(self, file):
        """
        Writes ESIpy's AOMs in AIMAll format in the working directory.

        Generates:
            - A '_atomicfiles/' directory with all the files created.
            - A '.int' file for each atom with its corresponding AOM.
            - A 'name.files' with a list of the names of the '.int' files.
            - A 'name.bad' with a standard input for the ESI-3D code.
            - For Natural Orbitals, a 'name.wfx' with the occupancies for the ESI-3D code.
        """

        for attr in ['mol', 'mf', 'aom', 'partition']:
            if getattr(self, attr, None) is None:
                raise AttributeError(
                    f" | Missing required attribute '{attr}'. Please define it before calling ESI.writeaoms")

        write_aoms(self.mol, self.mf, file, self.aom, self.rings, self.partition)
        print(f" | Written the AOMs in {self.readpath}/{file}_{self.partition}.aoms")

    def print(self):
        """
        Main output for ESIpy. Population analysis, localization and delocalization indices and
        electronic (and geometric) aromaticity indicators.
        """

        if self.molinfo is None or len(self.molinfo) == 1:
            if self.mol is None:
                raise ValueError(" | Missing 'mol' and/or 'molinfo'. Can not compute")
        if "calctype" not in self.molinfo:
            self.molinfo.update({"calctype": "Not specified"})
        if "xc" not in self.molinfo:
            self.molinfo.update({"xc": "Not specified"})
        if "energy" not in self.molinfo:
            self.molinfo.update({"energy": "Not specified"})
        if "method" not in self.molinfo:
            self.molinfo.update({"method": "Not specified"})

        if isinstance(self.partition, list):
            raise ValueError(
                " | Only one partition at a time. Partition should be a string, not a list\n | Please consider looping through the partitions before calling the function")

        if isinstance(self._aom, str):
            print(f" | Loading the AOMs from file {self._aom}")
            self._aom = self.aom
            if self.aom is None:
                raise NameError(" | Please provide a valid name to read the AOMs")

        if wf_type(self.aom) == "rest":
            from esipy.rest import info_rest, deloc_rest, arom_rest
            info_rest(self.totalaom, self.molinfo, self.nfrags)
            deloc_rest(self.totalaom, self.molinfo, self.fragmap)
            if self.rings:
                arom_rest(rings=self.rings, molinfo=self.molinfo, indicators=self.indicators, mci=self.mci,
                      av1245=self.av1245,
                      flurefs=self.flurefs, homarefs=self.homarefs, homerrefs=self.homerrefs, ncores=self.ncores, fragmap=self.fragmap)

        elif wf_type(self.aom) == "unrest":
            from esipy.unrest import info_unrest, deloc_unrest, arom_unrest
            info_unrest(self.totalaom, self.molinfo, self.nfrags)
            deloc_unrest(self.totalaom, self.molinfo, self.fragmap)
            if self.rings:
                arom_unrest(aom=self.aom, rings=self.rings, molinfo=self.molinfo, indicators=self.indicators, mci=self.mci,
                        av1245=self.av1245, partition=self.partition,
                        flurefs=self.flurefs, homarefs=self.homarefs, homerrefs=self.homerrefs, ncores=self.ncores, fragmap=self.fragmap,)

        elif wf_type(self.aom) == "no":
            from esipy.no import info_no, deloc_no, arom_no
            info_no(self.totalaom, self.molinfo, self.nfrags)
            deloc_no(self.totalaom, self.molinfo, self.fragmap)
            if self.rings:
                arom_no(rings=self.rings, molinfo=self.molinfo, indicators=self.indicators, mci=self.mci,
                    av1245=self.av1245,
                    flurefs=self.flurefs, homarefs=self.homarefs, homerrefs=self.homerrefs, ncores=self.ncores, fragmap=self.fragmap,)


environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"
environ["MKL_NUM_THREADS"] = "1"
environ["PYTHONBUFFERED"] = "1"
