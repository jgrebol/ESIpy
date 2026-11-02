import os
import unittest
from itertools import permutations

import numpy as np
from pyscf import gto, scf, dft, mp, mcscf

from esipy import ESI
from esipy.mci import compute_mci
from esipy.indicators import compute_iring, compute_iring_no, compute_homer
from esipy.tools import find_di, find_di_no, wf_type
from esipy.readfchk import readfchk

benzene = '''
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
'''
water = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
ring = [1, 2, 3, 4, 5, 6]

mol = gto.M(atom=benzene, basis='sto-3g', verbose=0)
rks = dft.RKS(mol, xc='B3LYP').run()
uks = dft.UKS(mol, xc='B3LYP').run()

props = ['iring', 'mci', 'pdi', 'av1245', 'flu', 'boa', 'boa_c']


def brute_mci(n, iring):
    return sum(iring(list(p)) for p in permutations(range(1, n + 1))) / (2 * n)


class ConsistencyTest(unittest.TestCase):

    def test_rks_uks_closed_shell(self):
        for part in ('mulliken', 'meta-lowdin', 'iao'):
            r = ESI(mol=mol, mf=rks, rings=ring, partition=part).indicators[0]
            u = ESI(mol=mol, mf=uks, rings=ring, partition=part).indicators[0]
            for p in props:
                self.assertAlmostEqual(getattr(r, p), getattr(u, p), places=5, msg=f"{part} {p}")

    def test_hf_as_natural_orbitals(self):
        for part in ('mulliken', 'lowdin'):
            esi = ESI(mol=mol, mf=rks, rings=ring, partition=part)
            r = esi.indicators[0]
            nos = [esi.aom, np.full(esi.aom[0].shape[0], 2.0)]
            no = ESI(aom=nos, mol=mol, mf=rks, rings=ring, partition=part, molinfo=esi.molinfo).indicators[0]
            for p in props:
                self.assertAlmostEqual(getattr(r, p), getattr(no, p), places=6, msg=f"{part} {p}")
            np.testing.assert_allclose(np.asarray(r.av1245_list, dtype=float), np.asarray(no.av1245_list, dtype=float), atol=1e-6)

    def test_mci_is_mean_of_irings(self):
        esi = ESI(mol=mol, mf=rks, rings=ring, partition='lowdin')
        aom = esi.aom
        occ = np.random.default_rng(0).uniform(0, 2, aom[0].shape[0])
        cases = {
            "rest": (aom, lambda a: 2 * compute_iring(a, aom), lambda: 2 * compute_mci(ring, aom, n_cores=1)),
            "no": ([aom, occ], lambda a: compute_iring_no(a, [aom, occ]), lambda: compute_mci(ring, [aom, occ], n_cores=1)),
        }
        for name, (a, iring, mci) in cases.items():
            self.assertAlmostEqual(mci(), brute_mci(6, iring), places=10, msg=name)

    def test_two_center_iring_is_di(self):
        esi = ESI(mol=mol, mf=rks, rings=ring, partition='lowdin')
        aom = esi.aom
        occ = np.random.default_rng(1).uniform(0, 2, aom[0].shape[0])
        self.assertAlmostEqual(2 * compute_iring([1, 2], aom), 2 * find_di(aom, 1, 2), places=10)
        self.assertAlmostEqual(compute_iring_no([1, 2], [aom, occ]), find_di_no([aom, occ], 1, 2), places=10)
        self.assertAlmostEqual(2 * find_di([aom, occ], 1, 2), find_di_no([aom, occ], 1, 2), places=10)


class MCITest(unittest.TestCase):

    def test_against_brute_force(self):
        esi_u = ESI(mol=mol, mf=uks, rings=ring, partition='lowdin')
        cas = ESI(mol=mol, mf=mcscf.CASSCF(scf.RHF(mol).run(), 6, 6).run(), rings=ring, partition='lowdin')
        for part in ('mulliken', 'lowdin', 'nao'):
            aom = ESI(mol=mol, mf=rks, rings=ring, partition=part).aom
            ref = brute_mci(6, lambda a: 2 * compute_iring(a, aom))
            for n_cores in (1, 2):
                self.assertAlmostEqual(2 * compute_mci(ring, aom, part, n_cores=n_cores), ref, places=10, msg=part)
        alpha = esi_u.aom[0]
        self.assertAlmostEqual(compute_mci(ring, alpha, n_cores=2), brute_mci(6, lambda a: compute_iring(a, alpha)), places=10)
        self.assertAlmostEqual(compute_mci(ring, cas.aom, n_cores=2), brute_mci(6, lambda a: compute_iring_no(a, cas.aom)), places=10)


class WavefunctionTest(unittest.TestCase):

    def test_iao_charged_and_odd_electrons(self):
        for atom, charge, spin, method in ((water, 1, 1, scf.UHF), ('O 0 0 0; H 0 0 0.97', 0, 1, scf.UHF),
                                           (water, 0, 0, scf.RHF)):
            m = gto.M(atom=atom, basis='6-31g', charge=charge, spin=spin, verbose=0)
            mf = method(m).run()
            aom = ESI(mol=m, mf=mf, partition='iao').aom
            if wf_type(aom) == "unrest":
                n = sum(np.trace(a) for spin_aom in aom for a in spin_aom)
            else:
                n = 2 * sum(np.trace(a) for a in aom)
            self.assertAlmostEqual(n, m.nelectron, places=6)

    def test_rohf_not_supported(self):
        m = gto.M(atom=water, basis='sto-3g', charge=1, spin=1, verbose=0)
        mf = scf.ROHF(m).run()
        with self.assertRaises(NotImplementedError):
            ESI(mol=m, mf=mf, partition='mulliken')

    def test_unrestricted_mp2_uses_total_density(self):
        m = gto.M(atom=water, basis='6-31g', verbose=0)
        rmp2 = mp.MP2(scf.RHF(m).run()).run()
        ump2 = mp.UMP2(scf.UHF(m).run()).run()
        r = ESI(mol=m, mf=rmp2, partition='mulliken').aom
        u = ESI(mol=m, mf=ump2, partition='mulliken').aom
        self.assertEqual(wf_type(u), "no")
        n = sum(np.sum(u[1] * np.diag(a)) for a in u[0])
        self.assertAlmostEqual(n, 10.0, places=6)
        self.assertAlmostEqual(find_di_no(r, 1, 2), find_di_no(u, 1, 2), places=5)

    def test_fchk_ump2_reads_correlated_density(self):
        path = os.path.join(os.path.dirname(__file__), 'FCHK', 'GAUSSIAN', '11_ump2.fchk')
        _, mf = readfchk(path)
        occ = np.asarray(mf.mo_occ)
        self.assertEqual(occ.ndim, 1)
        self.assertTrue(np.any((occ > 1e-3) & (np.abs(occ - 1) > 1e-3) & (np.abs(occ - 2) > 1e-3)))
        self.assertAlmostEqual(np.sum(occ), 16.0, places=4)


class GeometryTest(unittest.TestCase):

    def hexagon(self, r):
        ang = np.arange(6) * np.pi / 3
        return np.c_[r * np.cos(ang), r * np.sin(ang), np.zeros(6)] / 0.529177249

    def test_homer_reference_and_invariance(self):
        molinfo = {"symbols": ["C"] * 6, "geom": self.hexagon(1.437)}
        self.assertAlmostEqual(compute_homer(ring, molinfo), 1.0, places=6)

        geom = self.hexagon(1.40)
        geom[0] *= 1.05
        symbols = ["N", "C", "C", "C", "C", "C"]
        molinfo = {"symbols": symbols, "geom": geom}
        d = [np.linalg.norm(geom[i] - geom[(i + 1) % 6]) * 0.529177249 for i in range(6)]
        refs = {"CN": (506.43, 1.390), "CC": (950.74, 1.437)}
        bonds = ["CN", "CC", "CC", "CC", "CC", "CN"]
        expected = 1 - np.mean([refs[b][0] * (refs[b][1] - x) ** 2 for b, x in zip(bonds, d)])
        self.assertAlmostEqual(compute_homer(ring, molinfo), expected, places=8)
        for r in ([3, 4, 5, 6, 1, 2], [6, 5, 4, 3, 2, 1]):
            self.assertAlmostEqual(compute_homer(r, molinfo), expected, places=8)


if __name__ == '__main__':
    unittest.main()
