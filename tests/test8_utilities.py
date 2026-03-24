import unittest
import os
import numpy as np
from pyscf import gto, dft
import esipy
from esipy.readfchk import readfchk
from esipy.tools import find_rings, build_connectivity, format_partition, find_di

class TestUtilities(unittest.TestCase):
    def test_format_partition(self):
        self.assertEqual(format_partition("mulliken"), "mulliken")
        self.assertEqual(format_partition("meta-lowdin"), "meta-lowdin")
        self.assertEqual(format_partition("NAO"), "nao")

    def test_find_rings(self):
        # Connectivity for benzene-like ring
        connec = {
            1: [2, 6],
            2: [1, 3],
            3: [2, 4],
            4: [3, 5],
            5: [4, 6],
            6: [5, 1]
        }
        rings = find_rings(connec, minlen=6, maxlen=6)
        self.assertEqual(len(rings), 1)
        self.assertCountEqual(rings[0], [1, 2, 3, 4, 5, 6])

    def test_find_di(self):
        # Create dummy AOMs
        aom = [np.eye(2), np.zeros((2,2))]
        # di = 2 * Tr(A_i * A_j)
        # di(1,1) = 2 * Tr(I*I) = 2 * 2 = 4
        # di(1,2) = 2 * Tr(I*0) = 0
        self.assertEqual(find_di(aom, 1, 1), 4.0)
        self.assertEqual(find_di(aom, 1, 2), 0.0)

    def test_build_connectivity_benzene(self):
        mol = gto.M(
            atom='''
            C        0.000000000      0.000000000      1.393096000
            C        0.000000000      1.206457000      0.696548000
            C        0.000000000      1.206457000     -0.696548000
            C        0.000000000      0.000000000     -1.393096000
            C        0.000000000     -1.206457000     -0.696548000
            C        0.000000000     -1.206457000      0.696548000
            H        0.000000000      0.000000000      2.483127000
            H        0.000000000      2.150450000      1.241569000
            H        0.000000000      2.150450000     -1.241569000
            H        0.000000000      0.000000000     -2.483127000
            H        0.000000000     -2.150450000     -1.241569000
            H        0.000000000     -2.150450000      1.241569000
            ''',
            basis='sto-3g',
        )
        mf = dft.RKS(mol)
        mf.kernel()
        
        # Test build_connectivity with Mulliken
        esi = esipy.ESI(mol=mol, mf=mf, partition='mulliken')
        connec = build_connectivity(esi.aom)
        self.assertIsNotNone(connec)
        # C1 (index 1) should be connected to C2 (2) and C6 (6)
        self.assertIn(2, connec[1])
        self.assertIn(6, connec[1])

if __name__ == "__main__":
    unittest.main()
