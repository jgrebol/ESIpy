import unittest
from pyscf import gto, dft
from esipy import ESI

class TestActiveMCI(unittest.TestCase):
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
            ''', basis='sto-3g', spin=0, charge=0)
        cls.mol.build()
        cls.mf = dft.RKS(cls.mol)
        cls.mf.xc = 'B3LYP'
        cls.mf.kernel()
        cls.ring = [1, 2, 3, 4, 5, 6]

    def test_mci_approximations(self):
        # Initialize ESI object
        esi = ESI(mol=self.mol, mf=self.mf, rings=self.ring, partition='mulliken')
        
        # 1. Exact reference MCI (from indicators)
        inds = esi.indicators[0]
        exact_mci = inds.mci
        print(f"\nExact MCI (Mulliken): {exact_mci:.6f}")
        
        # 2. Test mciaprox with alg=5 (Odd Alternate) with d=3 (max distance for 6-ring)
        from esipy.mci import aproxmci
        val_5, nperms_5, t_5 = aproxmci(self.ring, esi.aom, esi.partition, alg=5, d=3, n_cores=1)
        val_5 *= 2.0
        print(f"Alg 5 MCI (d=3): {val_5:.6f} (perms: {nperms_5})")
        self.assertAlmostEqual(val_5, exact_mci, places=4)
        
        # 3. Test mciaprox with alg=6 (Active Connectivity DFS, delta=1e-4)
        val_6_exact, nperms_6_exact, t_6_exact = aproxmci(
            self.ring, esi.aom, esi.partition, alg=6, d=1, n_cores=1, delta=1e-4
        )
        val_6_exact *= 2.0
        print(f"Alg 6 MCI (delta=1e-4): {val_6_exact:.6f} (perms: {nperms_6_exact})")
        self.assertAlmostEqual(val_6_exact, exact_mci, places=5)
        self.assertEqual(nperms_6_exact, 120)  # All paths are connected at delta=1e-4

        # 4. Test mciaprox with alg=6 and delta=0.05 (some pruning)
        val_6_mid, nperms_6_mid, t_6_mid = aproxmci(
            self.ring, esi.aom, esi.partition, alg=6, d=1, n_cores=1, delta=0.05
        )
        val_6_mid *= 2.0
        print(f"Alg 6 MCI (delta=0.05): {val_6_mid:.6f} (perms: {nperms_6_mid})")
        # Ensure that it pruned some paths (nperms < 120)
        self.assertLess(nperms_6_mid, 120)

        # 5. Test mciaprox with alg=6 and delta=0.15 (aggressive pruning, only adjacent)
        val_6_agg, nperms_6_agg, t_6_agg = aproxmci(
            self.ring, esi.aom, esi.partition, alg=6, d=1, n_cores=1, delta=0.15
        )
        val_6_agg *= 2.0
        print(f"Alg 6 MCI (delta=0.15): {val_6_agg:.6f} (perms: {nperms_6_agg})")
        # For Benzene, delta=0.15 only keeps adjacent edges, so it only gets the 2 cycles (clockwise/counter-clockwise)
        self.assertEqual(nperms_6_agg, 2)

if __name__ == '__main__':
    unittest.main()
