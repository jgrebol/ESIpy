import unittest

from numpy import trace, dot
from pyscf import gto, dft

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
    ''', basis='sto-3g', spin=2, charge=0)
mol.build()
mf = dft.UKS(mol)
mf.xc = 'B3LYP'
mf.kernel()
ring = [1, 2, 3, 4, 5, 6]

# Expected results for different partitions
expected = {'mulliken': dict(exp_pop_atm1_a=3.1944, exp_pop_atm1_b=2.7968, exp_di_all_a=0.9840, exp_di_all_b=0.8946,
                             exp_dis_sum_a=7.2308, exp_dis_sum_b=7.2260, exp_di_12_a=0.7150, exp_di_12_b=0.5524,
                             exp_iring=0.007366, exp_mci=-0.023448, exp_av=1.165, exp_pdi=0.062509, exp_avmin=-3.501),
            'lowdin': dict(exp_pop_atm1_a=3.1460, exp_pop_atm1_b=2.7565, exp_di_all_a=0.9936, exp_di_all_b=0.9000,
                           exp_dis_sum_a=7.2857, exp_dis_sum_b=7.2612, exp_di_12_a=0.7227, exp_di_12_b=0.5574,
                           exp_iring=0.007371, exp_mci=-0.023151, exp_av=1.126, exp_pdi=0.062565, exp_avmin=-3.529),
            'meta_lowdin': dict(exp_pop_atm1_a=3.1486, exp_pop_atm1_b=2.7591, exp_di_all_a=0.9933, exp_di_all_b=0.8999,
                                exp_dis_sum_a=7.2831, exp_dis_sum_b=7.2600, exp_di_12_a=0.7226, exp_di_12_b=0.5575,
                                exp_iring=0.007380, exp_mci=-0.023148, exp_av=1.128, exp_pdi=0.062664,
                                exp_avmin=-3.527),
            'nao': dict(exp_pop_atm1_a=3.1291, exp_pop_atm1_b=2.7418, exp_di_all_a=0.9915, exp_di_all_b=0.8968,
                        exp_dis_sum_a=7.2669, exp_dis_sum_b=7.2487, exp_di_12_a=0.7205, exp_di_12_b=0.5554,
                        exp_iring=0.007423, exp_mci=-0.022291, exp_av=1.094, exp_pdi=0.062161, exp_avmin=-3.486),
            'iao': dict(exp_pop_atm1_a=3.1922, exp_pop_atm1_b=2.8019, exp_di_all_a=0.9852, exp_di_all_b=0.8925,
                        exp_dis_sum_a=7.2175, exp_dis_sum_b=7.2003, exp_di_12_a=0.7197, exp_di_12_b=0.5533,
                        exp_iring=0.007212, exp_mci=-0.023300, exp_av=1.102, exp_pdi=0.059074, exp_avmin=-3.559)}


class ESItest(unittest.TestCase):

    def run_pop_tests(self, partition, exp):
        esitest = ESI(mol=mol, mf=mf, rings=ring, partition=partition)
        aom = esitest.aom

        exp_pop_atm1_a = exp['exp_pop_atm1_a']
        exp_pop_atm1_b = exp['exp_pop_atm1_b']
        exp_di_all_a = exp['exp_di_all_a']
        exp_di_all_b = exp['exp_di_all_b']
        exp_dis_sum_a = exp['exp_dis_sum_a']
        exp_dis_sum_b = exp['exp_dis_sum_b']
        exp_di_12_a = exp['exp_di_12_a']
        exp_di_12_b = exp['exp_di_12_b']

        # Testing atomic populations of C1
        self.assertAlmostEqual(trace(aom[0][0]), exp_pop_atm1_a, places=3)
        self.assertAlmostEqual(trace(aom[1][0]), exp_pop_atm1_b, places=3)
        # Testing delocalized electrons for C1
        di_1_all_a = sum(trace(dot(aom[0][0], aom[0][i])) for i in range(1, 12))
        di_1_all_b = sum(trace(dot(aom[1][0], aom[1][i])) for i in range(1, 12))
        self.assertAlmostEqual(di_1_all_a, exp_di_all_a, places=3)
        self.assertAlmostEqual(di_1_all_b, exp_di_all_b, places=3)
        # Testing sum of DI and LI
        dis_a = [trace(dot(aom[0][i], aom[0][j])) for i in range(len(aom[0])) for j in range(len(aom[0])) if i != j]
        dis_b = [trace(dot(aom[1][i], aom[1][j])) for i in range(len(aom[1])) for j in range(len(aom[1])) if i != j]
        lis_a = [trace(dot(aom[0][i], aom[0][i])) for i in range(len(aom[0]))]
        lis_b = [trace(dot(aom[1][i], aom[1][i])) for i in range(len(aom[1]))]
        self.assertAlmostEqual(sum(dis_a), exp_dis_sum_a, places=3)
        self.assertAlmostEqual(sum(dis_b), exp_dis_sum_b, places=3)
        self.assertAlmostEqual(sum(lis_a) + sum(lis_b) + sum(dis_a) + sum(dis_b), 42, places=3)
        # Testing DI between C1 and C2
        self.assertAlmostEqual(2 * trace(dot(aom[0][0], aom[0][1])), exp_di_12_a, places=3)
        self.assertAlmostEqual(2 * trace(dot(aom[1][0], aom[1][1])), exp_di_12_b, places=3)

    def run_indicator_tests(self, partition, exp):
        esitest = ESI(mol=mol, mf=mf, rings=ring, partition=partition)
        inds = esitest.indicators[0]

        exp_iring = exp['exp_iring']
        exp_mci = exp['exp_mci']
        exp_av = exp['exp_av']
        exp_pdi = exp['exp_pdi']
        exp_avmin = exp['exp_avmin']

        # Multicenter indicators
        self.assertAlmostEqual(inds.iring, exp_iring, places=5)
        self.assertAlmostEqual(inds.mci, exp_mci, places=5)
        esitest = ESI(mol=mol, mf=mf, rings=ring, partition=partition, ncores=2)
        inds = esitest.indicators[0]
        self.assertAlmostEqual(inds.mci, exp_mci, places=5)
        self.assertAlmostEqual(inds.av1245, exp_av, places=2)
        self.assertAlmostEqual(inds.avmin, exp_avmin, places=2)
        self.assertAlmostEqual(inds.pdi, exp_pdi, places=5)

        # Geometric indicators
        self.assertAlmostEqual(inds.homa, 0.993307, places=5)
        self.assertAlmostEqual(inds.en, 0.006693, places=5)
        self.assertAlmostEqual(inds.geo, 0, places=5)
        self.assertAlmostEqual(inds.bla, 0, places=5)

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
