import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
from pyscf import scf, mcscf

sys.path.insert(0, os.path.expanduser("~/PycharmProjects/ESIpy"))
from esipy.readfchk import readfchk

base_dir = os.path.expanduser("~/PycharmProjects/ESIpy/joan/LiF/FCHK")
plot_dir = os.path.expanduser("~/PycharmProjects/ESIpy/joan/HARPOON/CALCS/plots/LiF")
os.makedirs(plot_dir, exist_ok=True)

def extract_distance(filename):
    basename = os.path.basename(filename)
    parts = basename.replace(".fchk", "").split("_")
    for p in reversed(parts):
        try:
            return float(p)
        except ValueError:
            pass
    return 0.0

files = glob.glob(os.path.join(base_dir, "lif_*.fchk"))
files.sort(key=extract_distance)

distances = []
e_fchk = []
e_pyscf_hf = []

print("=== Starting Profile Comparison ===")
for i, f_path in enumerate(files):
    dist = extract_distance(f_path)
    
    try:
        mol, mf_fchk = readfchk(f_path)
        e_f = mf_fchk.e_tot
    except Exception as e:
        print(f"Error reading {f_path}: {e}")
        continue
        
    distances.append(dist)
    e_fchk.append(e_f)
    
    # Run PySCF HF
    mol.verbose = 0
    mf_pyscf = scf.RHF(mol)
    mf_pyscf.kernel()
    e_pyscf_hf.append(mf_pyscf.e_tot)
    
    print(f"R = {dist:.1f} | E_FCHK = {e_f:.6f} | E_PySCF_HF = {mf_pyscf.e_tot:.6f}")
    
    # For exactly 3 selected points (small, min, large R), do a deeper check
    if i in [0, len(files)//2, len(files)-1]:
        print(f"  --> Deep check for R={dist:.1f}")
        # Run CASSCF(6,6) state-averaged as described in user's gen_lih.py
        try:
            mycas = mcscf.CASSCF(mf_pyscf, 6, 6)
            # SA-CASSCF 4 roots
            mycas = mycas.state_average([0.25, 0.25, 0.25, 0.25])
            mycas.kernel()
            print(f"      E_PySCF_SA_CASSCF = {mycas.e_tot:.6f}")
            
            # Compare density matrix traces
            dm_fchk = mf_fchk.make_rdm1()
            if dm_fchk is not None:
                tr_fchk = np.trace(dm_fchk @ mf_fchk.get_ovlp())
                print(f"      Trace DM FCHK = {tr_fchk:.4f}")
            else:
                print("      FCHK DM is None")
        except Exception as e:
            print(f"      CASSCF failed: {e}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(distances, e_fchk, 'b-o', label="FCHK Energy")
plt.plot(distances, e_pyscf_hf, 'r-x', label="PySCF RHF Energy")
plt.title("Potential Energy Profile: LiF")
plt.xlabel("Distance (Bohr/Angstrom)")
plt.ylabel("Total Energy (Hartree)")
plt.legend()
plt.grid(True)
plot_path = os.path.join(plot_dir, "Energy_Profile_Comparison.png")
plt.savefig(plot_path)
plt.close()

print(f"\nPlot saved to {plot_path}")
