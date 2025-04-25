import unittest

from numpy import trace, dot
from numpy.linalg import multi_dot
from pyscf import gto, scf, mcscf

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

mf = mcscf.CASSCF(myhf, 6, 6)
mf.kernel()

ring = [1, 2, 3, 4, 5, 6]

# Expected results for different partitions
expected = {
    'mulliken': dict(exp_pop_atm1=6.0614, exp_lif_atm1=4.2160, exp_lix_atm1=4.0724, exp_dif_1_all=1.8454,
                     exp_dix_1_all=1.9380,
                     exp_lifs_sum=27.9389, exp_lixs_sum=27.0775, exp_difs_sum=14.0611, exp_dixs_sum=14.6168,
                     exp_dif_12=1.3231,
                     exp_dix_12=1.3990, exp_iring=0.068999, exp_mci=0.097096, exp_av=10.971, exp_pdi=0.047143,
                     Nx=41.6943),
    'lowdin': dict(exp_pop_atm1=6.0299, exp_lif_atm1=4.1739, exp_lix_atm1=4.0304, exp_dif_1_all=1.8559,
                   exp_dix_1_all=1.9485,
                   exp_lifs_sum=27.8673, exp_lixs_sum=27.0059, exp_difs_sum=14.1327, exp_dixs_sum=14.6884,
                   exp_dif_12=1.3330,
                   exp_dix_12=1.4088, exp_iring=0.069018, exp_mci=0.097096, exp_av=10.976, exp_pdi=0.047430,
                   Nx=41.6943),
    'meta_lowdin': dict(exp_pop_atm1=6.0350, exp_lif_atm1=4.1795, exp_lix_atm1=4.0359, exp_dif_1_all=1.8556,
                        exp_dix_1_all=1.9482,
                        exp_lifs_sum=27.8704, exp_lixs_sum=27.0090, exp_difs_sum=14.1296, exp_dixs_sum=14.6854,
                        exp_dif_12=1.3329,
                        exp_dix_12=1.4088, exp_iring=0.069026, exp_mci=0.097098, exp_av=10.978, exp_pdi=0.047525,
                        Nx=41.6943),
    'nao': dict(exp_pop_atm1=6.0366, exp_lif_atm1=4.1847, exp_lix_atm1=4.0411, exp_dif_1_all=1.8519,
                exp_dix_1_all=1.9445,
                exp_lifs_sum=27.8929, exp_lixs_sum=27.0316, exp_difs_sum=14.1071, exp_dixs_sum=14.6628,
                exp_dif_12=1.3288,
                exp_dix_12=1.4047, exp_iring=0.069062, exp_mci=0.097121, exp_av=10.983, exp_pdi=0.047813, Nx=41.6943),
    'iao': dict(exp_pop_atm1=5.9580, exp_lif_atm1=4.1990, exp_lix_atm1=4.0555, exp_dif_1_all=1.7589,
                exp_dix_1_all=1.8343,
                exp_lifs_sum=28.0713, exp_lixs_sum=27.2100, exp_difs_sum=13.5524, exp_dixs_sum=13.8516,
                exp_dif_12=1.2418,
                exp_dix_12=1.3177, exp_iring=0.069071, exp_mci=0.097161, exp_av=11.002, exp_pdi=0.048389, Nx=41.0615)
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
            self.assertAlmostEqual(sum(lifs) + sum(difs), 41.3672, places=4)
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

    def test_nao(self):
        partition = 'nao'
        exp = expected[partition]
        self.run_pop_tests(partition, exp=exp)
        self.run_indicator_tests(partition, exp=exp)

    def test_iao(self):
        partition = 'iao'
        exp = expected[partition]
        self.run_pop_tests(partition, exp=exp)
        self.run_indicator_tests(partition, exp=exp)


if __name__ == "__main__":
    unittest.main()
