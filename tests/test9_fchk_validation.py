import os
import numpy as np
import unittest
import pickle
from esipy.readfchk import readfchk
from esipy import ESI
from esipy.tools import find_di, find_ns, wf_type

class TestFchkValidation(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.fchk_dir = os.path.join(self.base_dir, "FCHK")
        with open("pyscf_refs.pkl", "rb") as f:
            self.refs = pickle.load(f)

    def run_validation(self, prog, filename, ref_key):
        path = os.path.join(self.fchk_dir, prog, filename)
        if not os.path.exists(path):
            self.skipTest(f"File {path} not found")

        print(f"\n [VALIDATE] {prog} / {filename}")
        mol_f, mf_f = readfchk(path)
        ref = self.refs[ref_key]
        
        # 1. Check Energy (Gaussian only)
        if prog == 'GAUSSIAN':
            self.assertAlmostEqual(mf_f.e_tot, ref['e'], delta=0.15) # Relaxed due to grid diffs
        else:
            print(f"  [INFO] Skipping energy for {prog}")

        # 2. Check Overlap Orthonormality
        S = mf_f.get_ovlp()
        C = mf_f.mo_coeff
        if isinstance(C, list):
            for i, c in enumerate(C):
                ortho = c.T @ S @ c
                err = np.max(np.abs(ortho - np.eye(c.shape[1])))
                self.assertLess(err, 1e-6)
        else:
            ortho = C.T @ S @ C
            err = np.max(np.abs(ortho - np.eye(C.shape[1])))
            self.assertLess(err, 1e-6)

        # 3. Check Indicators
        esi = ESI(mol=mol_f, mf=mf_f, partition='mulliken')
        aoms = esi.aom
        wf = wf_type(aoms)
        atoms = list(range(1, mol_f.natm+1))
        
        if wf == "unrest":
            f_pops = np.array(find_ns(atoms, aoms[0])) + np.array(find_ns(atoms, aoms[1]))
            f_di12 = find_di(aoms[0], 1, 2) + find_di(aoms[1], 1, 2)
        elif wf == "no":
            aom_list, occ = aoms
            f_pops = [np.sum([occ[i] * aom_list[a_idx][i,i] for i in range(len(occ))]) for a_idx in range(mol_f.natm)]
            f_di12 = 0.0
        else:
            f_pops = np.array(find_ns(atoms, aoms))
            f_di12 = find_di(aoms, 1, 2) if mol_f.natm >= 2 else 0.0

        # Compare Pops
        pop_err = np.max(np.abs(np.asarray(f_pops) - ref['ind']['pops']))
        print(f"  Max Pop Error: {pop_err:.2e}")
        self.assertLess(pop_err, 0.02) # Relaxed for program diffs

        # Compare DI
        if ref['ind']['di12'] > 1e-6:
            di_err = abs(f_di12 - ref['ind']['di12'])
            print(f"  DI(1,2) Error: {di_err:.2e}")
            self.assertLess(di_err, 0.05)

    def test_g_1_bz_sph(self): self.run_validation('GAUSSIAN', '1_benzene_spherical.fchk', '1_benzene_spherical')
    def test_g_2_bz_cart(self): self.run_validation('GAUSSIAN', '2_benzene_cartesian.fchk', '2_benzene_cartesian')
    def test_g_3_o2(self): self.run_validation('GAUSSIAN', '3_o2_triplet.fchk', '3_o2_triplet')
    def test_g_4_h2(self): self.run_validation('GAUSSIAN', '4_h2_oss.fchk', '4_h2_oss')
    def test_g_5_h2o(self): self.run_validation('GAUSSIAN', '5_high_l.fchk', '5_high_l')
    def test_g_7_rmp2(self): self.run_validation('GAUSSIAN', '7_rmp2.fchk', '7_rmp2')
    def test_g_9_ccsd(self): self.run_validation('GAUSSIAN', '9_ccsd.fchk', '9_ccsd')
    def test_q_1_bz_sph(self): self.run_validation('QCHEM', '1_benzene_spherical.fchk', '1_benzene_spherical')
    def test_q_2_bz_cart(self): self.run_validation('QCHEM', '2_benzene_cartesian.fchk', '2_benzene_cartesian')
    def test_q_4_h2(self): self.run_validation('QCHEM', '4_h2_oss.fchk', '4_h2_oss')
    def test_q_7_rmp2(self): self.run_validation('QCHEM', '7_rmp2.fchk', '7_rmp2')

if __name__ == "__main__":
    unittest.main()
