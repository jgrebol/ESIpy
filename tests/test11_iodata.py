#!/usr/bin/env python3
import os
import unittest
import numpy as np

from esipy.parser import convert_to_fchk, parse_and_read
from esipy.readfchk import readfchk
from esipy.input import ESIInput
from esipy import ESI


class TestIODataIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(os.path.dirname(__file__), "IODATA")
        cls.generated_fchks = []

    @classmethod
    def tearDownClass(cls):
        for fchk_file in cls.generated_fchks:
            if os.path.isfile(fchk_file):
                try:
                    os.remove(fchk_file)
                except OSError:
                    pass

    def track_fchk(self, fchk_path):
        if fchk_path not in self.generated_fchks:
            self.generated_fchks.append(fchk_path)
        return fchk_path

    def test_bz_molden_benzene(self):
        """Tests reading an ORCA Molden file directly via readfchk and calculating sensitive indicators (DI, PDI, Iring, MCI)."""
        molden_path = os.path.join(self.data_dir, "bz.molden")
        expected_fchk = os.path.join(self.data_dir, "bz.esipy.fchk")
        self.track_fchk(expected_fchk)

        mol, mf = readfchk(molden_path)
        self.assertEqual(mol.natm, 12)
        self.assertEqual(mol.nao, 192)

        esi = ESI(mol=mol, mf=mf, rings=[[1, 2, 3, 4, 5, 6]], partition="meta-lowdin")
        self.assertIsNotNone(esi.aom)
        aom_a, aom_b = esi.aom
        pop_tot = sum(np.trace(a) for a in aom_a) + sum(np.trace(b) for b in aom_b)
        self.assertAlmostEqual(pop_tot, 42.0, places=4)

        # Sensitive indicator checks on benzene ring
        ind = esi.indicators[0]
        self.assertAlmostEqual(ind.iring, 0.08628, places=3)
        self.assertAlmostEqual(ind.mci, 0.12891, places=3)
        self.assertAlmostEqual(ind.pdi, 0.11722, places=3)
        self.assertAlmostEqual(ind.homa, 0.96782, places=3)

        # Delocalization index (DI) checks
        di_c1_c2 = 2.0 * (np.trace(aom_a[0] @ aom_a[1]) + np.trace(aom_b[0] @ aom_b[1]))
        di_c1_c4 = 2.0 * (np.trace(aom_a[0] @ aom_a[3]) + np.trace(aom_b[0] @ aom_b[3]))
        self.assertAlmostEqual(di_c1_c2, 1.4391, places=3)
        self.assertAlmostEqual(di_c1_c4, 0.1172, places=3)

    def test_nh3_orca_molden(self):
        """Tests converting and reading an ORCA Molden file and checking N-H DI."""
        orca_path = os.path.join(self.data_dir, "nh3_orca.molden")
        expected_fchk = os.path.join(self.data_dir, "nh3_orca.esipy.fchk")
        self.track_fchk(expected_fchk)

        mol, mf = readfchk(orca_path)
        self.assertEqual(mol.natm, 4)
        self.assertEqual(mol.nao, 50)

        esi = ESI(mol=mol, mf=mf, partition="meta-lowdin", rings=None)
        pop_tot = sum(2.0 * np.trace(a) for a in esi.aom)
        self.assertAlmostEqual(pop_tot, 10.0, places=2)

        # Sensitive DI check: covalent N-H single bond DI ~ 0.94
        di_n_h = 4.0 * np.trace(esi.aom[0] @ esi.aom[1])
        self.assertAlmostEqual(di_n_h, 0.937, places=2)

    def test_li2_molekel_unrestricted(self):
        """Tests converting and reading an unrestricted Molekel (.mkl) file."""
        mkl_path = os.path.join(self.data_dir, "li2.mkl")
        expected_fchk = os.path.join(self.data_dir, "li2.esipy.fchk")
        self.track_fchk(expected_fchk)

        mol, mf = readfchk(mkl_path)
        self.assertEqual(mol.natm, 2)
        self.assertEqual(mol.nao, 110)

        esi = ESI(mol=mol, mf=mf, partition="mulliken", rings=None)
        aom_a, aom_b = esi.aom
        pop_a = sum(np.trace(a) for a in aom_a)
        pop_b = sum(np.trace(b) for b in aom_b)
        self.assertAlmostEqual(pop_a, 3.0, places=4)
        self.assertAlmostEqual(pop_b, 2.0, places=4)
        self.assertAlmostEqual(pop_a + pop_b, 5.0, places=4)

    def test_h2_wfx_unrestricted(self):
        """Tests converting and reading an unrestricted AIM WFX file and checking H-H DI."""
        wfx_path = os.path.join(self.data_dir, "h2_ub3lyp_ccpvtz.wfx")
        expected_fchk = os.path.join(self.data_dir, "h2_ub3lyp_ccpvtz.esipy.fchk")
        self.track_fchk(expected_fchk)

        mol, mf = readfchk(wfx_path)
        self.assertEqual(mol.natm, 2)
        self.assertEqual(mol.nao, 34)

        esi = ESI(mol=mol, mf=mf, partition="mulliken", rings=None)
        aom_a, aom_b = esi.aom
        pop_a = sum(np.trace(a) for a in aom_a)
        pop_b = sum(np.trace(b) for b in aom_b)
        self.assertAlmostEqual(pop_a, 1.0, places=4)
        self.assertAlmostEqual(pop_b, 1.0, places=4)
        self.assertAlmostEqual(pop_a + pop_b, 2.0, places=4)

        # Sensitive DI check: H-H bond
        di_h_h = 2.0 * (np.trace(aom_a[0] @ aom_a[1]) + np.trace(aom_b[0] @ aom_b[1]))
        self.assertAlmostEqual(di_h_h, 1.0, places=3)

    def test_ethanol_molekel(self):
        """Tests converting and reading a Molekel file for ethanol."""
        mkl_path = os.path.join(self.data_dir, "ethanol.mkl")
        expected_fchk = os.path.join(self.data_dir, "ethanol.esipy.fchk")
        self.track_fchk(expected_fchk)

        mol, mf = readfchk(mkl_path)
        self.assertEqual(mol.natm, 9)
        self.assertEqual(mol.nao, 39)

        esi = ESI(mol=mol, mf=mf, partition="mulliken", rings=None)
        pop_tot = sum(2.0 * np.trace(a) for a in esi.aom)
        self.assertAlmostEqual(pop_tot, 26.0, places=2)

    def test_h2o_sto3g_wfn(self):
        """Tests converting and reading an AIM WFN file for water."""
        wfn_path = os.path.join(self.data_dir, "h2o_sto3g.wfn")
        expected_fchk = os.path.join(self.data_dir, "h2o_sto3g.esipy.fchk")
        self.track_fchk(expected_fchk)

        mol, mf = readfchk(wfn_path)
        self.assertEqual(mol.natm, 3)
        self.assertEqual(mol.nao, 21)

        esi = ESI(mol=mol, mf=mf, partition="mulliken", rings=None)
        pop_tot = sum(2.0 * np.trace(a) for a in esi.aom)
        self.assertAlmostEqual(pop_tot, 10.0, places=2)

    def test_nh3_psi4_molden(self):
        """Tests converting and reading a PSI4 Molden file for ammonia."""
        molden_path = os.path.join(self.data_dir, "nh3_psi4.molden")
        expected_fchk = os.path.join(self.data_dir, "nh3_psi4.esipy.fchk")
        self.track_fchk(expected_fchk)

        mol, mf = readfchk(molden_path)
        self.assertEqual(mol.natm, 4)
        self.assertEqual(mol.nao, 50)

        esi = ESI(mol=mol, mf=mf, partition="mulliken", rings=None)
        pop_tot = sum(2.0 * np.trace(a) for a in esi.aom)
        self.assertAlmostEqual(pop_tot, 10.0, places=2)

    def test_nh3_turbomole_molden(self):
        """Tests converting and reading a Turbomole Molden file."""
        molden_path = os.path.join(self.data_dir, "nh3_turbomole.molden")
        expected_fchk = os.path.join(self.data_dir, "nh3_turbomole.esipy.fchk")
        self.track_fchk(expected_fchk)

        mol, mf = readfchk(molden_path)
        self.assertEqual(mol.natm, 4)
        self.assertEqual(mol.nao, 52)

        esi = ESI(mol=mol, mf=mf, partition="mulliken", rings=None)
        pop_tot = sum(2.0 * np.trace(a) for a in esi.aom)
        self.assertAlmostEqual(pop_tot, 10.0, places=2)

    def test_nh3_molpro_molden(self):
        """Tests converting and reading a Molpro Molden file."""
        molden_path = os.path.join(self.data_dir, "nh3_molpro2012.molden")
        expected_fchk = os.path.join(self.data_dir, "nh3_molpro2012.esipy.fchk")
        self.track_fchk(expected_fchk)

        mol, mf = readfchk(molden_path)
        self.assertEqual(mol.natm, 4)
        self.assertEqual(mol.nao, 52)

        esi = ESI(mol=mol, mf=mf, partition="mulliken", rings=None)
        pop_tot = sum(2.0 * np.trace(a) for a in esi.aom)
        self.assertAlmostEqual(pop_tot, 10.0, places=2)

    def test_h2o_psi4_cartesian_molden(self):
        """Tests converting and reading a Cartesian PSI4 Molden file."""
        molden_path = os.path.join(self.data_dir, "h2o_psi4_1.3.2_6-31G_d_cart.molden")
        expected_fchk = os.path.join(self.data_dir, "h2o_psi4_1.3.2_6-31G_d_cart.esipy.fchk")
        self.track_fchk(expected_fchk)

        mol, mf = readfchk(molden_path)
        self.assertEqual(mol.natm, 3)
        self.assertEqual(mol.nao, 19)

        esi = ESI(mol=mol, mf=mf, partition="mulliken", rings=None)
        pop_tot = sum(2.0 * np.trace(a) for a in esi.aom)
        self.assertAlmostEqual(pop_tot, 10.0, places=2)

    def test_water_sto3g_wfx(self):
        """Tests converting and reading an AIM WFX file for water."""
        wfx_path = os.path.join(self.data_dir, "water_sto3g_hf.wfx")
        expected_fchk = os.path.join(self.data_dir, "water_sto3g_hf.esipy.fchk")
        self.track_fchk(expected_fchk)

        mol, mf = readfchk(wfx_path)
        self.assertEqual(mol.natm, 3)
        self.assertEqual(mol.nao, 21)

        esi = ESI(mol=mol, mf=mf, partition="mulliken", rings=None)
        pop_tot = sum(2.0 * np.trace(a) for a in esi.aom)
        self.assertAlmostEqual(pop_tot, 10.0, places=2)

    def test_o2_uhf_wfn_unrestricted(self):
        """Tests converting and reading an unrestricted triplet O2 AIM WFN file."""
        wfn_path = os.path.join(self.data_dir, "o2_uhf.wfn")
        expected_fchk = os.path.join(self.data_dir, "o2_uhf.esipy.fchk")
        self.track_fchk(expected_fchk)

        mol, mf = readfchk(wfn_path)
        self.assertEqual(mol.natm, 2)
        self.assertEqual(mol.nao, 72)

        esi = ESI(mol=mol, mf=mf, partition="mulliken", rings=None)
        aom_a, aom_b = esi.aom
        pop_a = sum(np.trace(a) for a in aom_a)
        pop_b = sum(np.trace(b) for b in aom_b)
        self.assertAlmostEqual(pop_a, 9.0, places=4)
        self.assertAlmostEqual(pop_b, 7.0, places=4)
        self.assertAlmostEqual(pop_a + pop_b, 16.0, places=4)

    def test_h2o_cfour_molden_read(self):
        """Tests converting and reading a CFOUR Molden file."""
        molden_path = os.path.join(self.data_dir, "h2o_ccpvdz_cfour.molden")
        expected_fchk = os.path.join(self.data_dir, "h2o_ccpvdz_cfour.esipy.fchk")
        self.track_fchk(expected_fchk)

        mol, mf = readfchk(molden_path)
        self.assertEqual(mol.natm, 1)
        self.assertEqual(mol.nao, 15)

    def test_ecp_with_readfile_input(self):
        """Tests parsing a .hs input with $READFILE and $ECP."""
        fchk_path = os.path.join(self.data_dir, "monosilicic_acid_hf_lan.fchk")
        hs_content = f"""
$READFILE
{fchk_path}
$ECP
lanl2dz
$PARTITION
meta-lowdin
"""
        inp = ESIInput.from_string(hs_content)
        self.assertEqual(inp.mode, "readfile")
        self.assertEqual(inp.fchk_file, fchk_path)
        self.assertEqual(inp.ecp, "lanl2dz")

        mol, mf = readfchk(inp.fchk_file, ecp=inp.ecp)
        self.assertEqual(mol.natm, 9)
        self.assertEqual(mol.ecp, "lanl2dz")
        self.assertEqual(mol.nao, 28)

    def test_conversion_error_on_log_and_out_files(self):
        """Tests that attempting to convert log and out files without MO data raises errors."""
        bad_files = [
            "water_orca.out",
            "water_sto3g_hf_g03.log",
            "water_hf_ccpvtz_freq_qchem.out",
            "carbon_gs_ae_contracted.cp2k.out",
            "orca_gradient.out",
        ]
        for bf in bad_files:
            path = os.path.join(self.data_dir, bf)
            with self.subTest(file=bf):
                with self.assertRaises(Exception):
                    convert_to_fchk(path)

    def test_file_not_found_error(self):
        """Tests that a non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            convert_to_fchk("non_existent_file.molden")


if __name__ == "__main__":
    unittest.main()
