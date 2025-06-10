import unittest

from numpy import trace, dot
from numpy.linalg import multi_dot
from pyscf import gto, scf, cc

from esipy import ESI

# Set up molecule and DFT calculations
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
    ''', basis='sto-3g', spin=0, charge=0)
mol.build()

myhf = scf.RHF(mol)
myhf.kernel()

mf = cc.CCSD(myhf)
mf.kernel()

ring = [1, 2, 3, 4, 5, 6]

# Expected results for different partitions
expected = {
    'mulliken': dict(exp_pop_atm1=6.0599, exp_lif_atm1=4.5839, exp_lix_atm1=4.0714, exp_dif_1_all=1.4760,
                     exp_dix_1_all=1.8358,
                     exp_lifs_sum=30.7821, exp_lixs_sum=27.0794, exp_difs_sum=11.2179, exp_dixs_sum=13.8702,
                     exp_dif_12=1.0579,
                     exp_dix_12=1.3204, exp_iring=0.051591, exp_mci=0.073406, exp_av=9.540, exp_pdi=0.049767,
                     Nx=40.9496),
    'lowdin': dict(exp_pop_atm1=6.0291, exp_lif_atm1=4.5438, exp_lix_atm1=4.0295, exp_dif_1_all=1.4853,
                   exp_dix_1_all=1.8466,
                   exp_lifs_sum=30.7179, exp_lixs_sum=27.0052, exp_difs_sum=11.2821, exp_dixs_sum=13.9444,
                   exp_dif_12=1.0662,
                   exp_dix_12=1.3304, exp_iring=0.051584, exp_mci=0.073380, exp_av=9.539, exp_pdi=0.049823, Nx=40.9496),
    'meta_lowdin': dict(exp_pop_atm1=6.0342, exp_lif_atm1=4.5493, exp_lix_atm1=4.0351, exp_dif_1_all=1.4849,
                        exp_dix_1_all=1.8461,
                        exp_lifs_sum=30.7211, exp_lixs_sum=27.0088, exp_difs_sum=11.2789, exp_dixs_sum=13.9408,
                        exp_dif_12=1.0662,
                        exp_dix_12=1.3304, exp_iring=0.051593, exp_mci=0.073383, exp_av=9.540, exp_pdi=0.049897,
                        Nx=40.9496),
    'nao': dict(exp_pop_atm1=6.6990, exp_lif_atm1=5.3656, exp_lix_atm1=4.8778, exp_dif_1_all=1.3345,
                exp_dix_1_all=1.6670,
                exp_lifs_sum=32.5293, exp_lixs_sum=29.1924, exp_difs_sum=9.4707, exp_dixs_sum=11.7572,
                exp_dif_12=0.9546,
                exp_dix_12=1.1976, exp_iring=0.049725, exp_mci=0.071585, exp_av=9.221, exp_pdi=0.047375, Nx=40.9496),
    'iao': dict(exp_pop_atm1=5.96295, exp_lif_atm1=4.6204, exp_lix_atm1=4.0622, exp_dif_1_all=1.3425,
                exp_dix_1_all=1.7355,
                exp_lifs_sum=31.1848, exp_lixs_sum=27.2483, exp_difs_sum=10.4666, exp_dixs_sum=13.1295,
                exp_dif_12=0.9896,
                exp_dix_12=1.2421, exp_iring=0.051636, exp_mci=0.0734550, exp_av=9.561, exp_pdi=0.049779, Nx=40.3779)
}


class ESItest(unittest.TestCase):

    def run_pop_tests(self, partition, exp):
        esitest = ESI(mol=mol, mf=mf, myhf=myhf, rings=ring, partition=partition)
        esitest.print()
        aom, occ = esitest.aom

        exp_pop_atm1 = exp['exp_pop_atm1']
        exp_lif_atm1 = exp['exp_lif_atm1']
        exp_lix_atm1 = exp['exp_lix_atm1']
        exp_dif_1_all = exp['exp_dif_1_all']
        exp_dix_1_all = exp['exp_dix_1_all']
        exp_lifs_sum = exp['exp_lifs_sum']
        exp_lixs_sum = exp['exp_lixs_sum']
        exp_difs_sum = exp['exp_difs_sum']
        exp_dixs_sum = exp['exp_dixs_sum']
        exp_dif_12 = exp['exp_dif_12']
        exp_dix_12 = exp['exp_dix_12']
        Nx = exp['Nx']

        # Testing atomic populations of C1
        self.assertAlmostEqual(trace(dot(occ, aom[0])), exp_pop_atm1, places=4)
        # Testing LI for C1
        lif = trace(multi_dot((occ ** (1 / 2), aom[0], occ ** (1 / 2), aom[0])))
        lix = 0.5 * trace(multi_dot((occ, aom[0], occ, aom[0])))
        self.assertAlmostEqual(lif, exp_lif_atm1, places=4)
        self.assertAlmostEqual(lix, exp_lix_atm1, places=4)
        # Testing delocalized electrons for C1
        dif_1_all = trace(dot(occ, aom[0])) - lif
        # dix_1_all = trace(dot(occ, aom[0])) - lix
        dix_1_all = 0.5 * sum(trace(multi_dot((occ, aom[0], occ, aom[i]))) for i in range(1, 12))
        self.assertAlmostEqual(dif_1_all, exp_dif_1_all, places=4)
        self.assertAlmostEqual(dix_1_all, exp_dix_1_all, places=4)
        # Testing sum of DI and LI
        lifs = [trace(multi_dot((occ ** (1 / 2), aom[i], occ ** (1 / 2), aom[i]))) for i in range(len(aom))]
        lixs = [0.5 * trace(multi_dot((occ, aom[i], occ, aom[i]))) for i in range(len(aom))]
        difs = [trace(multi_dot((occ ** (1 / 2), aom[i], occ ** (1 / 2), aom[j]))) for i in range(len(aom)) for j in
                range(len(aom)) if i != j]
        dixs = [0.5 * trace(multi_dot((occ, aom[i], occ, aom[j]))) for i in range(len(aom)) for j in range(len(aom)) if
                i != j]
        N = sum([trace(dot(occ, aom[i])) for i in range(0, 12)])
        self.assertAlmostEqual(sum(lifs), exp_lifs_sum, places=4)
        self.assertAlmostEqual(sum(lixs), exp_lixs_sum, places=4)
        self.assertAlmostEqual(N - sum(lifs), exp_difs_sum, places=4)
        self.assertAlmostEqual(sum(dixs), exp_dixs_sum, places=4)
        if partition == "iao":
            self.assertAlmostEqual(sum(lifs) + sum(difs), 41.7481, places=4)
        else:
            self.assertAlmostEqual(sum(lifs) + sum(difs), 42, places=4)

        self.assertAlmostEqual(sum(lixs) + sum(dixs), Nx, places=4)

        # Testing DI between C1 and C2
        self.assertAlmostEqual(2 * trace(multi_dot((occ ** (1 / 2), aom[0], occ ** (1 / 2), aom[1]))), exp_dif_12,
                               places=4)
        self.assertAlmostEqual(trace(multi_dot((occ, aom[0], occ, aom[1]))), exp_dix_12, places=4)

    def run_indicator_tests(self, partition, exp):
        esitest = ESI(mol=mol, mf=mf, myhf=myhf, rings=ring, partition=partition)
        inds = esitest.indicators[0]

        exp_iring = exp['exp_iring']
        exp_mci = exp['exp_mci']
        exp_av = exp['exp_av']
        exp_pdi = exp['exp_pdi']

        # Multicenter indicators
        self.assertAlmostEqual(inds.iring, exp_iring, places=6)
        self.assertAlmostEqual(inds.mci, exp_mci, places=5)
        esitest = ESI(mol=mol, mf=mf, myhf=myhf, rings=ring, partition=partition, ncores=2)
        inds = esitest.indicators[0]
        self.assertAlmostEqual(inds.mci, exp_mci, places=5)
        self.assertAlmostEqual(inds.av1245, exp_av, places=2)
        self.assertAlmostEqual(inds.pdi, exp_pdi, places=6)

        # Geometric indicators
        self.assertAlmostEqual(inds.homa, 0.993307, places=6)
        self.assertAlmostEqual(inds.en, 0.006693, places=6)
        self.assertAlmostEqual(inds.geo, 0, places=6)
        self.assertAlmostEqual(inds.bla, 0, places=6)

    def test_mulliken(self):
        partition = 'mulliken'
        exp = expected[partition]
        self.run_pop_tests(partition, exp=exp)
        self.run_indicator_tests(partition, exp=exp)

    def test_lowdin(self):
        partition = 'lowdin'
        exp = expected[partition]
        self.run_pop_tests(partition, exp=exp)
        self.run_indicator_tests(partition, exp=exp)

    def test_meta_lowdin(self):
        partition = 'meta_lowdin'
        exp = expected[partition]
        self.run_pop_tests(partition, exp=exp)
        self.run_indicator_tests(partition, exp=exp)

    # PySCF should be installed from source to avoid Issue #1755, solved in commit #1803.
    # For simplicity, this test is removed as the bug is not corrected through updating with pip.
    # def test_nao(self):
    #    partition = 'nao'
    #    exp = expected[partition]
    #    self.run_pop_tests(partition, exp=exp)
    #    self.run_indicator_tests(partition, exp=exp)

    def test_iao(self):
        partition = 'iao'
        exp = expected[partition]
        self.run_pop_tests(partition, exp=exp)
        self.run_indicator_tests(partition, exp=exp)


if __name__ == "__main__":
    unittest.main()
