
import os
import sys
import subprocess
import numpy as np

def run_cmd(cmd, env=None):
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"FAILED: {cmd}")
        print(result.stdout)
        print(result.stderr)
        return False
    print("SUCCESS")
    return True

def test_fchk(branch_path, fchk_path, label):
    print(f"\n--- Testing FCHK {label}: {fchk_path} ---")
    script = f"""
import sys
import os
sys.path.insert(0, '{branch_path}')
import numpy as np
from esipy.readfchk import Mole2, MeanField2
try:
    mol = Mole2('{fchk_path}')
    mf = MeanField2('{fchk_path}', mol)
    print(f"Branch: {os.path.basename(branch_path)}")
    print(f"Label: {label}")
    print(f"Calculated type: {{mf.__name__}}")
    if hasattr(mf, 'mo_occ'):
        if isinstance(mf.mo_occ, list):
            print(f"Alpha occ shape: {{np.shape(mf.mo_occ[0])}}")
            print(f"Beta occ shape: {{np.shape(mf.mo_occ[1])}}")
        else:
            print(f"Occ shape: {{np.shape(mf.mo_occ)}}")
    
    # Natural Orbitals check
    if hasattr(mf, 'mo_occ') and not isinstance(mf.mo_occ, list):
        occ = mf.mo_occ
        is_natorb = np.any((occ > 1e-6) & (np.abs(occ - 1.0) > 1e-6) & (np.abs(occ - 2.0) > 1e-6))
        print(f"Is Natural Orbital: {{is_natorb}}")
    
    # Q-Chem check
    if hasattr(mol.fchk, 'is_qchem'):
        print(f"Is Q-Chem: {{mol.fchk.is_qchem}}")
        if mol.fchk.is_qchem:
            S = mf._scf.get_ovlp()
            print(f"Overlap matrix shape: {{S.shape}}")
            rdm1 = mf.make_rdm1()
            if isinstance(rdm1, np.ndarray) and rdm1.ndim == 3:
                 pop = np.trace(rdm1[0] @ S) + np.trace(rdm1[1] @ S)
            else:
                 pop = np.trace(rdm1 @ S)
            print(f"Total population: {{pop}}")
except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    with open("temp_test.py", "w") as f:
        f.write(script)
    
    env = os.environ.copy()
    env["PYTHONPATH"] = branch_path
    return run_cmd("python3 temp_test.py", env=env)

def run_all_tests(branch_path):
    print(f"\n=== Running All Tests for {branch_path} ===")
    test_dir = f"{branch_path}/tests"
    env = os.environ.copy()
    env["PYTHONPATH"] = branch_path
    
    all_passed = True
    for f in sorted(os.listdir(test_dir)):
        if f.startswith("test") and f.endswith(".py") and "test6" not in f:
            print(f"Running {f}...")
            # We must run from the branch root to ensure it picks up the correct package
            if not run_cmd(f"PYTHONPATH={branch_path} python3 {test_dir}/{f}", env=env):
                all_passed = False
    return all_passed

if len(sys.argv) > 1:
    path_arg = sys.argv[1]
    if os.path.isdir(path_arg):
        branch_path = os.path.abspath(path_arg)
        branch = os.path.basename(branch_path)
    else:
        branch = path_arg
        branch_path = os.path.abspath(f"../branch_verification/{branch}")
else:
    print("Usage: python3 verify_all.py <branch_name_or_path>")
    sys.exit(1)

success = True
success &= run_all_tests(branch_path)

# Wavefunction Reading Tests
success &= test_fchk(branch_path, f"{branch_path}/FCHK/GAUSSIAN/h2o_sto3g.fchk", "Restricted")
success &= test_fchk(branch_path, f"{branch_path}/FCHK/GAUSSIAN/bzt.fchk", "Unrestricted")
success &= test_fchk(branch_path, f"{branch_path}/FCHK/GAUSSIAN/lih_cas.fchk", "Natural Orbitals")
success &= test_fchk(branch_path, f"{branch_path}/FCHK/QCHEM/h2o_sto3g.fchk", "Q-Chem")

if success:
    print(f"\nDONE: All checks PASSED for {branch}")
else:
    print(f"\nDONE: Some checks FAILED for {branch}")
