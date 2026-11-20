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

    def test_find_rings_monocycle_without_atom1(self):
        # Zn6O6 ring around a central atom 1 with no bonds (Cl@Zn6O6-): every ring atom has two neighbours
        ring = [2, 9, 3, 8, 7, 13, 6, 12, 5, 11, 4, 10]
        connec = {a: [ring[i - 1], ring[(i + 1) % len(ring)]] for i, a in enumerate(ring)}
        rings = find_rings(connec, minlen=6, maxlen=12)
        self.assertEqual(len(rings), 1)
        self.assertCountEqual(rings[0], ring)

    def test_iao_reference_basis(self):
        from pyscf import scf
        from pyscf.lo import iao, orth
        mol = gto.M(atom='O 0 0 0.1193; H 0 0.7632 -0.4770; H 0 -0.7632 -0.4770', basis='6-31g', verbose=0)
        mf = scf.RHF(mol).run()
        S = mol.intor('int1e_ovlp')
        C = mf.mo_coeff[:, mf.mo_occ > 0]
        aoms = {}
        for ref in ('minao', 'sto-3g'):
            esi = esipy.ESI(mol=mol, mf=mf, partition='iao' if ref == 'minao' else f'iao {ref}')  # dev-iao: basis in the partition
            U = orth.vec_lowdin(iao.iao(mol, C, minao=ref), S)
            pmol = mol.copy(); pmol.build(False, False, basis=ref)
            direct = [C.T @ S @ U[:, a:b] @ U[:, a:b].T @ S @ C for a, b in pmol.aoslice_by_atom()[:, 2:]]
            for x, y in zip(esi.aom, direct):
                np.testing.assert_allclose(x, y, atol=1e-10)
            aoms[ref] = esi.aom
        self.assertGreater(abs(np.trace(aoms['minao'][0]) - np.trace(aoms['sto-3g'][0])), 1e-4)

    def test_input_ringthres_iaoref(self):
        from esipy.input import ESIInput
        inp = ESIInput.from_string("$READFCHK\nx.fchk\n$RINGTHRES\n0.2\n$RING\n1 2 3 4 5 6\n$IAOREF\nsto-3g\n")
        self.assertAlmostEqual(inp.rings_thres, 0.2)
        self.assertEqual(inp.iaoref, 'sto-3g')
        self.assertEqual(inp.rings, [[1, 2, 3, 4, 5, 6]])  # $RINGTHRES is not taken as a $RING block
        self.assertEqual(ESIInput.from_string("$READFCHK\nx.fchk\n").iaoref, 'minao')

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
