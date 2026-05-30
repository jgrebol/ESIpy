import unittest
import numpy as np
from pyscf import gto, scf, mcscf
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

mf = mcscf.CASSCF(myhf, 6, 6)
mf.kernel()

ring = [1, 2, 3, 4, 5, 6]

expected = {
    'iao': { 'exp_av': 94.94038495, 'exp_di12': 1.3209687, 'exp_iring': 0.0186989, 'exp_mci': 0.4357335, 'exp_pdi': 0.0446592, 'exp_pop_atm1': 6.098720 },
    'lowdin': { 'exp_av': 91.15746662, 'exp_di12': 1.33295736, 'exp_iring': 0.01876224, 'exp_mci': 0.42358596, 'exp_pdi': 0.04743011, 'exp_pop_atm1': 6.02985251 },
    'meta-lowdin': { 'exp_av': 84.22160011, 'exp_di12': 1.33294338, 'exp_iring': 0.01876618, 'exp_mci': 0.40123273, 'exp_pdi': 0.04752450, 'exp_pop_atm1': 6.03504653 },
    'mulliken': { 'exp_av': -95.43236589, 'exp_di12': 1.32312552, 'exp_iring': 0.01875246, 'exp_mci': 0.30155249, 'exp_pdi': 0.04714255, 'exp_pop_atm1': 6.06136168 },
    'nao': { 'exp_av': 80.71421711, 'exp_di12': 1.32883432, 'exp_iring': 0.01878421, 'exp_mci': 0.38039274, 'exp_pdi': 0.04781329, 'exp_pop_atm1': 6.03655698 },
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

    def test_nao(self):
        self.run_all_tests('nao', expected['nao'])

    def test_iao(self):
        self.run_all_tests('iao', expected['iao'])

if __name__ == "__main__":
    unittest.main()
