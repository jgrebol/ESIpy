import unittest
import os
import numpy as np
from pyscf import gto, scf, mp, mcscf
import pyscf.lo as lo

from esipy.readfchk import readfchk
from esipy.make_aoms import make_aoms
from esipy import ESI
import esipy.iao as iao_mod
from esipy.tools import wf_type

def get_q_di(aom):
    wf = wf_type(aom)
    if wf == "rest":
        Q = [2 * np.trace(a) for a in aom]
    elif wf == "unrest":
        aom_a, aom_b = aom
        Q = [np.trace(aom_a[i]) + np.trace(aom_b[i]) for i in range(len(aom_a))]
    elif wf == "no":
        aom_list, occ = aom
        if occ.ndim == 2: occ = np.diag(occ)
        Q = [np.sum(occ * np.diag(a)) for a in aom_list]
    return Q, None

class TestAdvancedIAO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build basic test molecules
        cls.mol_rhf = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='cc-pVDZ', verbose=0)
        cls.mf_rhf = scf.RHF(cls.mol_rhf).run()
        
        cls.mol_uhf = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='cc-pVDZ', spin=1, charge=1, verbose=0)
        cls.mf_uhf = scf.UHF(cls.mol_uhf).run()

        cls.mf_mp2 = mp.MP2(cls.mf_rhf).run()
        
        cls.mf_cas = mcscf.CASSCF(cls.mf_rhf, 2, 2).run()
        
        cls.fchk_dir = os.path.join(os.path.dirname(__file__), 'FCHK')
        cls.bz_fchk = os.path.join(cls.fchk_dir, 'GAUSSIAN', 'bz.fchk')
        
        # A list of almost all partitions to test universally
        cls.all_partitions = [
            'iao', 'iao2', 'iao-autosad', 'iao-autosad2',
            'iao-effao-net', 'iao-effao-gross', 'iao-effao-lowdin', 
            'iao-effao-ml', 'iao-effao-nao', 'iao-effao-sym', 
            'iao-effao-sps', 'iao-effao-spsa',
            'fpiao', 'fpiao2', 'fpiao(2.0)', 'fpiao2(1.5)',
            'dfpiao(0.6)', 'dfpiao2(0.3)', 'peiao', 'peiao gross', 'dpeiao(0.2)', 'wiao'
        ]

    def check_total_traces(self, mol, mf, expected_n, exact=True):
        """Verifies that all partitions conserve the total number of electrons."""
        for p in self.all_partitions:
            with self.subTest(partition=p):
                esi = ESI(mol=mol, mf=mf, partition=p)
                Q, _ = get_q_di(esi.aom)
                N_tot = sum(Q)
                if exact:
                    self.assertAlmostEqual(N_tot, expected_n, places=3, 
                                           msg=f"Total electrons failed for {p}")
                else:
                    self.assertAlmostEqual(N_tot, expected_n, delta=0.05, 
                                           msg=f"Total electrons failed for {p}")

    def test_total_traces_rhf(self):
        self.check_total_traces(self.mol_rhf, self.mf_rhf, 4.0, exact=True)
        
    def test_total_traces_uhf(self):
        self.check_total_traces(self.mol_uhf, self.mf_uhf, 3.0, exact=True)

    def test_total_traces_mp2(self):
        self.check_total_traces(self.mol_rhf, self.mf_mp2, 4.0, exact=False)

    def test_total_traces_cas(self):
        self.check_total_traces(self.mol_rhf, self.mf_cas, 4.0, exact=False)

        
    def test_total_traces_fchk_bz(self):
        if not os.path.exists(self.bz_fchk):
            self.skipTest('bz.fchk not found')
        mol, mf = readfchk(self.bz_fchk)
        # We test a subset for speed
        partitions = ['iao', 'iao-effao-nao', 'fpiao(2.0)', 'dfpiao(0.6)', 'peiao']
        for p in partitions:
            with self.subTest(partition=p):
                esi = ESI(mol=mol, mf=mf, partition=p)
                Q, _ = get_q_di(esi.aom)
                N_tot = sum(Q)
                self.assertAlmostEqual(N_tot, 42.0, places=3)

    def test_fpiao_exponents_scaling(self):
        """
        Verify that FPIAO(x) scales the exponents of the polarization functions.
        """
        mol, mf = self.mol_rhf, self.mf_rhf
        
        # Direct call to get pmol
        _, pmol_1 = iao_mod.fpiao(mol, mf.mo_coeff, x=1.0)
        _, pmol_2 = iao_mod.fpiao(mol, mf.mo_coeff, x=2.0)
        
        # Li atom is index 0. We find its basis in pmol
        basis_1 = pmol_1._basis['Li']
        basis_2 = pmol_2._basis['Li']
        
        # minao part should be identical
        self.assertEqual(len(basis_1), len(basis_2))
        
        # The last shell is the polarization shell (p-shell)
        pol_shell_1 = basis_1[-1]
        pol_shell_2 = basis_2[-1]
        
        self.assertEqual(pol_shell_1[0], 1) # angular momentum p
        
        # The exponent of the first primitive should be scaled by 2.0
        exp_1 = pol_shell_1[1][0]
        exp_2 = pol_shell_2[1][0]
        self.assertAlmostEqual(exp_2, exp_1 * 2.0, places=8)

    def test_blending_dfpiao(self):
        mol, mf = self.mol_rhf, self.mf_rhf
        p = 0.6
        aom_iao = make_aoms(mol, mf, 'iao')
        aom_fpiao = make_aoms(mol, mf, 'fpiao(1.0)')
        aom_dfpiao = make_aoms(mol, mf, f'dfpiao({p})')
        
        for i in range(mol.natm):
            expected = p * aom_iao[i] + (1.0 - p) * aom_fpiao[i]
            np.testing.assert_allclose(aom_dfpiao[i], expected, atol=1e-8)

    def test_blending_dfpiao2(self):
        mol, mf = self.mol_rhf, self.mf_rhf
        p = 0.6
        aom_iao2 = make_aoms(mol, mf, 'iao2')
        aom_fpiao2 = make_aoms(mol, mf, 'fpiao2(1.0)')
        aom_dfpiao2 = make_aoms(mol, mf, f'dfpiao2({p})')
        
        for i in range(mol.natm):
            expected = p * aom_iao2[i] + (1.0 - p) * aom_fpiao2[i]
            np.testing.assert_allclose(aom_dfpiao2[i], expected, atol=1e-8)

    def test_blending_dpeiao(self):
        mol, mf = self.mol_rhf, self.mf_rhf
        p = 0.2
        aom_iao = make_aoms(mol, mf, 'iao-effao-nao')
        aom_peiao = make_aoms(mol, mf, 'peiao')
        aom_dpeiao = make_aoms(mol, mf, f'dpeiao({p})')
        
        for i in range(mol.natm):
            expected = p * aom_iao[i] + (1.0 - p) * aom_peiao[i]
            np.testing.assert_allclose(aom_dpeiao[i], expected, atol=1e-8)

if __name__ == '__main__':
    unittest.main()
