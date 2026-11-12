import unittest
import itertools
import numpy as np
from pyscf import gto, dft
from esipy import ESI
from esipy.mci import compute_mci, compute_mci_dfs, mci_approx, mci_approx_dfs, _glynn_sum


class TestGlynnMCI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
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
            ''', basis='sto-3g', spin=0, charge=0, verbose=0)
        cls.mf = dft.RKS(cls.mol)
        cls.mf.xc = 'B3LYP'
        cls.mf.kernel()
        cls.ring = [1, 2, 3, 4, 5, 6]

    def test_brute_force(self):
        # Non-symmetric low-rank matrices and a directed adjacency, against explicit enumeration
        rng = np.random.default_rng(0)
        n = 7
        mats = [rng.normal(size=(6, 3)) @ rng.normal(size=(3, 6)) for _ in range(n)]
        adj = rng.random((n, n)) < 0.6
        np.fill_diagonal(adj, False)
        ref_all, ref_adj, count = 0.0, 0.0, 0
        for p in itertools.permutations(range(1, n)):
            tr = np.trace(np.linalg.multi_dot([mats[0]] + [mats[i] for i in p]))
            ref_all += tr
            cycle = (0,) + p + (0,)
            if all(adj[cycle[i], cycle[i + 1]] for i in range(n)):
                ref_adj += tr
                count += 1
        val, _ = _glynn_sum(mats, None, 1, 64)
        self.assertAlmostEqual(val / ref_all, 1.0, places=10)
        for n_cores, batch_mem in ((1, 64), (4, 1e-5)):
            val, perms = _glynn_sum(mats, adj, n_cores, batch_mem)
            self.assertAlmostEqual(val / ref_adj, 1.0, places=10)
            self.assertEqual(perms, count)

    def test_against_dfs(self):
        for partition in ('mulliken', 'meta-lowdin'):
            esi = ESI(mol=self.mol, mf=self.mf, rings=self.ring, partition=partition)
            aom, connec = esi.aom, esi.connec
            self.assertAlmostEqual(compute_mci(self.ring, aom, partition, n_cores=1),
                                   compute_mci_dfs(self.ring, aom, partition, n_cores=1), places=12)
            cases = [dict(alg=0)] + [dict(alg=a, d=d) for a in (1, 2, 3, 4, 5) for d in (1, 2, 3)]
            cases += [dict(alg=6, delta=delta) for delta in (1e-4, 0.05, 0.15)]
            for kw in cases:
                v, p, _ = mci_approx_dfs(self.ring, aom, partition, n_cores=1, connec=connec, **kw)
                g, q, _ = mci_approx(self.ring, aom, partition, n_cores=1, connec=connec, **kw)
                self.assertAlmostEqual(v, g, places=12, msg=f"{partition} {kw}")
                self.assertEqual(p, q, msg=f"{partition} {kw}")


if __name__ == '__main__':
    unittest.main()
