"""
test9_1_fchk_script.py
Mirrors test9_fchk_validation.py but exercises the esipy CLI script
instead of calling the ESI Python API directly.

For every FCHK file listed in pyscf_refs.pkl it:
  1. Builds a .hs input file with the appropriate $READFCHK, $PARTITION,
     $RING (and optional extras) blocks.
  2. Runs the esipy script in a subprocess.
  3. Asserts the exit code is 0.
  4. Parses atomic populations N(Sij) and DI(1,2) from stdout.
  5. Compares them against the reference pickle (same tolerances as test9).

Additional input-parser features exercised per file:
  1_benzene_spherical  -> $AV1245
  2_benzene_cartesian  -> $FLUREF (custom FLU reference)
  3_o2_triplet         -> $FRAGMENTS + $NOMCI
  4_h2_oss             -> $DOMCI
  6_ecp                -> $ECP lanl2dz
  8_cisd               -> $NCORES 2
"""

import os
import re
import sys
import pickle
import subprocess
import tempfile
import unittest
import numpy as np


class TestFchkScript(unittest.TestCase):

    # ------------------------------------------------------------------
    # setUp / helpers
    # ------------------------------------------------------------------
    def setUp(self):
        self.base_dir  = os.path.dirname(__file__)
        self.fchk_dir  = os.path.join(self.base_dir, "FCHK", "GAUSSIAN")
        self.script    = os.path.abspath(
            os.path.join(self.base_dir, "..", "scripts", "esipy"))
        self.pkl_path  = os.path.join(self.base_dir, "pyscf_refs.pkl")

        if not os.path.exists(self.pkl_path):
            self.skipTest(
                "pyscf_refs.pkl is missing. Run tests/generate_refs.py first.")

        import sys
        if hasattr(np, '_core'):
            sys.modules['numpy._core'] = np._core
        else:
            try:
                import numpy.core as np_core
                sys.modules['numpy._core'] = np_core
            except ImportError:
                pass
        with open(self.pkl_path, "rb") as fh:
            self.refs = pickle.load(fh)


        # Map: ref_key -> (fchk_filename, ring_atoms, extra_blocks)
        self.cases = {
            "1_benzene_spherical": (
                "1_benzene_spherical.fchk",
                [1, 2, 3, 4, 5, 6],
                "$AV1245\n"),
            "2_benzene_cartesian": (
                "2_benzene_cartesian.fchk",
                [1, 2, 3, 4, 5, 6],
                "$FLUREF\n1.0 1.0\n"),
            "3_o2_triplet": (
                "3_o2_triplet.fchk",
                [1, 2],
                "$FRAGMENTS\n1\n2\n$NOMCI\n"),
            "4_h2_oss": (
                "4_h2_oss.fchk",
                [1, 2],
                "$DOMCI\n"),
            "5_high_l": (
                "5_high_l.fchk",
                [1, 2, 3],
                ""),
            "6_ecp": (
                "6_ecp.fchk",
                [1, 2],
                "$ECP\nlanl2dz\n"),
            "7_rmp2": (
                "7_rmp2.fchk",
                [1, 2, 3],
                ""),
            "8_cisd": (
                "8_cisd.fchk",
                [1, 2, 3],
                "$NCORES\n2\n"),
            "9_ccsd": (
                "9_ccsd.fchk",
                [1, 2, 3],
                ""),
            "10_casscf_rest": (
                "10_casscf_rest.fchk",
                [1, 2],
                ""),
            "11_ump2": (
                "11_ump2.fchk",
                [1, 2],
                ""),
            "12_casscf_unrest": (
                "12_casscf_unrest.fchk",
                [1, 2],
                ""),
            "13_anthracene": (
                "13_anthracene.fchk",
                [1, 2, 3, 4, 12, 10, 13, 5, 6, 7, 8, 14, 9, 11],
                ""),
        }

    # ------------------------------------------------------------------
    def _build_input(self, fchk_path, ring_atoms, extra_blocks):
        """Return a .hs input string."""
        ring_str = " ".join(str(a) for a in ring_atoms)
        content  = f"$READFCHK\n{fchk_path}\n"
        content += "$PARTITION\nmulliken\n"
        content += f"$RING\n{ring_str}\n"
        content += extra_blocks
        return content

    def _run_script(self, inp_content):
        """Write a temp .hs file, run the esipy script, return (rc, stdout, stderr)."""
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath(
            os.path.join(self.base_dir, ".."))

        with tempfile.NamedTemporaryFile(
                "w", delete=False, suffix=".hs") as tmp:
            tmp.write(inp_content)
            inp_path = tmp.name

        try:
            result = subprocess.run(
                [sys.executable, self.script, inp_path],
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
            )
            return result.returncode, result.stdout, result.stderr
        finally:
            if os.path.exists(inp_path):
                os.remove(inp_path)

    # ------------------------------------------------------------------
    # Output parsers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_pops(stdout):
        """
        Parse N(Sij) column from atom lines in both restricted and unrestricted
        output formats:
          Restricted  : |  C 1    6.1034    4.1811    1.9223         (3 floats)
          Unrestricted: |  O 1    8.0000    4.5000    3.5000  0.2049  0.7172  (5 floats)
        Returns a numpy array of populations (one per atom, in order).
        """
        pops = []
        # Match atom lines: symbol + number, then 3 or 5 floats; capture first float.
        pat = re.compile(
            r"^\s*\|\s+[A-Z][a-z]?\s*\d+\s+([\d.+-]+)(?:\s+[\d.+-]+){2,}\s*$")
        for line in stdout.splitlines():
            m = pat.match(line)
            if m:
                pops.append(float(m.group(1)))
        return np.array(pops)

    @staticmethod
    def _parse_di12(stdout):
        """
        Parse DI(1,2) from atom-pair lines in both restricted and unrestricted
        output formats:
          Restricted  : |  C1 - C2      1.3693               (1 float)
          Unrestricted: |  O1 - O2      1.8443  0.4099  1.4344  (3 floats)
        Returns the first float (total DI) or None if not found.
        The printed DI equals 2 * find_di (the pkl reference stores find_di).
        """
        # Match atom-pair lines where the first atom index is 1 and second is 2
        pat = re.compile(
            r"^\s*\|\s+[A-Z][a-z]?\s*1\s*-\s*[A-Z][a-z]?\s*2\s+([\d.+-]+)")
        for line in stdout.splitlines():
            m = pat.match(line)
            if m:
                return float(m.group(1))
        return None

    # ------------------------------------------------------------------
    # One combined test that loops all cases (mirrors test9 structure)
    # ------------------------------------------------------------------
    def _run_one(self, ref_key):
        if ref_key not in self.refs:
            self.skipTest(f"Reference for {ref_key} missing in pkl")

        fchk_file, ring_atoms, extra = self.cases[ref_key]
        fchk_path = os.path.join(self.fchk_dir, fchk_file)
        if not os.path.exists(fchk_path):
            self.skipTest(f"FCHK file not found: {fchk_path}")

        ref = self.refs[ref_key]

        inp = self._build_input(fchk_path, ring_atoms, extra)
        rc, stdout, stderr = self._run_script(inp)

        # 1. Exit code
        self.assertEqual(
            rc, 0,
            f"[{ref_key}] esipy exited with code {rc}.\nSTDERR:\n{stderr}")

        # 2. Populations
        pops = self._parse_pops(stdout)
        self.assertGreater(
            len(pops), 0,
            f"[{ref_key}] Could not parse any N(Sij) populations from output.\n"
            f"STDOUT:\n{stdout[:2000]}")

        ref_pops = np.asarray(ref["ind"]["pops"])
        # ref_pops may only cover a subset of atoms (ring atoms used in generate_refs)
        n = min(len(pops), len(ref_pops))
        pop_err = np.max(np.abs(pops[:n] - ref_pops[:n]))
        self.assertLess(
            pop_err, 0.02,
            f"[{ref_key}] Max population error {pop_err:.4f} >= 0.02\n"
            f"  got:      {pops[:n]}\n"
            f"  expected: {ref_pops[:n]}")

        # 3. DI(1,2) — only when a meaningful reference value exists.
        # NOTE: the pkl stores find_di(aoms, 1, 2) which is half the printed DI
        # (the script prints 2*S_ij*S_ji while find_di sums only one direction).
        # We halve the parsed value before comparing.
        ref_di = ref["ind"].get("di12", 0.0)
        if ref_di is not None and float(ref_di) > 1e-6:
            di_parsed = self._parse_di12(stdout)
            if di_parsed is not None:
                di_half = di_parsed / 2.0
                di_err = abs(di_half - float(ref_di))
                self.assertLess(
                    di_err, 0.05,
                    f"[{ref_key}] DI(1,2) error {di_err:.4f} >= 0.05\n"
                    f"  got (halved): {di_half}  expected: {ref_di}")

        # 4. Extra assertions for specific input-parser features
        if ref_key == "1_benzene_spherical":
            self.assertIn("AV1245", stdout,
                          f"[{ref_key}] AV1245 block not found in output")
            self.assertIn("Using the default FLU references", stdout,
                          f"[{ref_key}] Did not print that default FLU references were used")
        if ref_key == "2_benzene_cartesian":
            self.assertIn("Using FLU references provided by the user", stdout,
                          f"[{ref_key}] Did not print that custom FLU references were used")
        if ref_key == "6_ecp":
            self.assertIn("Applying ECP: lanl2dz", stdout,
                          f"[{ref_key}] ECP print statement not found in output")
        if ref_key == "3_o2_triplet":
            self.assertNotIn("MCI", stdout,
                             f"[{ref_key}] MCI should be suppressed by $NOMCI")

    # ------------------------------------------------------------------
    # One test method per case (same style as test9)
    # ------------------------------------------------------------------
    def test_1_benzene_spherical(self):
        self._run_one("1_benzene_spherical")

    def test_2_benzene_cartesian(self):
        self._run_one("2_benzene_cartesian")

    def test_3_o2_triplet(self):
        self._run_one("3_o2_triplet")

    def test_4_h2_oss(self):
        self._run_one("4_h2_oss")

    def test_5_high_l(self):
        self._run_one("5_high_l")

    def test_6_ecp(self):
        self._run_one("6_ecp")

    def test_7_rmp2(self):
        self._run_one("7_rmp2")

    def test_8_cisd(self):
        self._run_one("8_cisd")

    def test_9_ccsd(self):
        self._run_one("9_ccsd")

    def test_10_casscf_rest(self):
        self._run_one("10_casscf_rest")

    def test_11_ump2(self):
        self._run_one("11_ump2")

    def test_12_casscf_unrest(self):
        self._run_one("12_casscf_unrest")

    def test_13_anthracene(self):
        self._run_one("13_anthracene")

    def test_2_benzene_cartesian_default_flu(self):
        """Compare FLU value of benzene under default and custom reference settings."""
        # 1. Run with custom reference
        fchk_file, ring_atoms, extra = self.cases["2_benzene_cartesian"]
        fchk_path = os.path.join(self.fchk_dir, fchk_file)
        inp_custom = self._build_input(fchk_path, ring_atoms, extra)
        rc_custom, stdout_custom, _ = self._run_script(inp_custom)
        self.assertEqual(rc_custom, 0)
        
        flu_custom_match = re.search(r"\|\s*FLU\s+\d+\s*=\s*([\d.e+-]+)", stdout_custom)
        self.assertIsNotNone(flu_custom_match, "Could not find FLU in custom run stdout")
        flu_custom = float(flu_custom_match.group(1))

        # 2. Run with default references
        inp_default = self._build_input(fchk_path, ring_atoms, "")
        rc_default, stdout_default, _ = self._run_script(inp_default)
        self.assertEqual(rc_default, 0)

        flu_default_match = re.search(r"\|\s*FLU\s+\d+\s*=\s*([\d.e+-]+)", stdout_default)
        self.assertIsNotNone(flu_default_match, "Could not find FLU in default run stdout")
        flu_default = float(flu_default_match.group(1))

        print(f"\n  [FLU COMPARE] Default: {flu_default:.6f}  Custom: {flu_custom:.6f}")
        # The values should be distinctly different because custom references (1.0 1.0)
        # differ greatly from standard aromatic reference values.
        self.assertNotAlmostEqual(flu_default, flu_custom, places=4)

    def test_1_benzene_spherical_findrings(self):
        """Test $FINDRINGS keyword on benzene."""
        fchk_file = "1_benzene_spherical.fchk"
        fchk_path = os.path.join(self.fchk_dir, fchk_file)
        # Construct input with $FINDRINGS and no $RING block
        content  = f"$READFCHK\n{fchk_path}\n"
        content += "$PARTITION\nmulliken\n"
        content += "$FINDRINGS\n"
        
        rc, stdout, stderr = self._run_script(content)
        self.assertEqual(rc, 0, f"esipy exited with code {rc}.\nSTDERR:\n{stderr}")
        self.assertIn("Ring  1 (6):   1  2  3  4  5  6", stdout)


if __name__ == "__main__":
    unittest.main()
