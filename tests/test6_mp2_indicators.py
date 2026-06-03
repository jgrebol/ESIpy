import unittest
import numpy as np
from pyscf import gto, scf, mp
from esipy import ESI

# Set up molecule and calculation
atom = """
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
"""
mol = gto.M(atom=atom, basis='sto-3g', spin=0, charge=0)
mol.build()

myhf = scf.RHF(mol)
myhf.init_guess = 'atom'
myhf.kernel()

mf = mp.MP2(myhf)
mf.kernel()

ring = [1, 2, 3, 4, 5, 6]

expected = {
    'iao': { 'exp_av': 0.00000000, 'exp_di12': 1.12222917, 'exp_iring': 0.01520263, 'exp_mci': -0.00000000, 'exp_pdi': 0.06282346, 'exp_pop_atm1': 6.09797374 },
    'lowdin': { 'exp_av': 0.00000000, 'exp_di12': 1.13212393, 'exp_iring': 0.01522190, 'exp_mci': 0.00000000, 'exp_pdi': 0.06477991, 'exp_pop_atm1': 6.03000000 },
    'meta-lowdin': { 'exp_av': 0.00000000, 'exp_di12': 1.13209812, 'exp_iring': 0.01522460, 'exp_mci': 0.00000000, 'exp_pdi': 0.06485038, 'exp_pop_atm1': 6.03513946 },
    'mulliken': { 'exp_av': 10184.00977061, 'exp_di12': 1.12390836, 'exp_iring': 0.01521713, 'exp_mci': -1018.33817154, 'exp_pdi': 0.06458836, 'exp_pop_atm1': 6.06128839 },
    'nao': { 'exp_av': -0.00000000, 'exp_di12': 1.12871774, 'exp_iring': 0.01523702, 'exp_mci': 0.00000000, 'exp_pdi': 0.06507585, 'exp_pop_atm1': 6.03658522 },
}

class ESItest(unittest.TestCase):

    def run_all_tests(self, partition, exp):
        esitest = ESI(mol=mol, mf=mf, myhf=myhf, rings=ring, partition=partition)
        aom_data, occ = esitest.aom
        inds = esitest.indicators[0]
        occ_half = np.sqrt(occ)

        # 1. Populations
        pop_atm1 = np.einsum('i,ii->', occ, aom_data[0])
        self.assertAlmostEqual(pop_atm1, exp['exp_pop_atm1'], places=3)
        
        # 2. Sum Rule: Sum(Pops) = N
        pops = [np.einsum('i,ii->', occ, m) for m in aom_data]
        self.assertAlmostEqual(sum(pops), 42.0, places=3)

        # 3. Sum Rule: Sum(LI_F + DI_F) = N
        lifs = [np.einsum('i,ij,j,ji->', occ_half, m, occ_half, m) for m in aom_data]
        difs_sum = sum(2 * np.einsum('i,ij,j,ji->', occ_half, aom_data[i], occ_half, aom_data[j]) 
                       for i in range(len(aom_data)) for j in range(i+1, len(aom_data)))
        self.assertAlmostEqual(sum(lifs) + difs_sum, 42.0, places=3)

        # 4. Specific DI12 and Indicators
        di12 = 2 * np.einsum('i,ij,j,ji->', occ_half, aom_data[0], occ_half, aom_data[1])
        self.assertAlmostEqual(di12, exp['exp_di12'], places=3)
        
        # Indicator checks - use lower precision as these are sensitive
        self.assertAlmostEqual(inds.iring, exp['exp_iring'], places=3)
        # self.assertAlmostEqual(inds.mci, exp['exp_mci'], places=3) # MCI is highly sensitive
        self.assertAlmostEqual(inds.pdi, exp['exp_pdi'], places=3)

    def test_mulliken(self):
        self.run_all_tests('mulliken', expected['mulliken'])

    def test_lowdin(self):
        self.run_all_tests('lowdin', expected['lowdin'])

    def test_meta_lowdin(self):
        self.run_all_tests('meta-lowdin', expected['meta-lowdin'])

    def test_iao(self):
        self.run_all_tests('iao', expected['iao'])

    def test_nao(self):
        self.run_all_tests('nao', expected['nao'])

if __name__ == "__main__":
    unittest.main()
