import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

try:
    with open('joan/HARPOON/LiF/chk/LiF_data.pkl', 'rb') as f:
        d_lif, e_lif, data_lif = pickle.load(f)
    print(f"LiF points: {len(d_lif)}")
    if len(d_lif) > 1:
        plt.figure(figsize=(8,5))
        for fam in data_lif:
            # Match keys properly
            keys = list(data_lif[fam].keys())
            y = [data_lif[fam][k] for k in keys]
            plt.plot(keys, y, 'o-', label=fam)
        plt.xlabel('Distance R (Å)')
        plt.ylabel('DI (Exact)')
        plt.title('LiF Current DIs (Sequential PySCF)')
        plt.legend()
        plt.grid(True)
        plt.savefig('/home/joan/.gemini/antigravity-cli/brain/e97fce9c-463e-4f5b-8f3b-7cb078d54c5d/LiF_Current_DIs.png')
        print("Generated LiF_Current_DIs.png")
except Exception as e:
    print(f"LiF Error: {e}")

try:
    with open('joan/LiH/GS/chk/LiH-GS_data.pkl', 'rb') as f:
        d_lih, e_lih, data_lih = pickle.load(f)
    print(f"LiH GS points: {len(d_lih)}")
    if len(d_lih) > 1:
        plt.figure(figsize=(8,5))
        for fam in data_lih:
            keys = sorted(list(data_lih[fam].keys()))
            y = [data_lih[fam][k] for k in keys]
            plt.plot(keys, y, 'o-', label=fam)
        plt.xlabel('Distance R (Å)')
        plt.ylabel('DI (Exact)')
        plt.title('LiH GS Current DIs (FCI)')
        plt.legend()
        plt.grid(True)
        plt.savefig('/home/joan/.gemini/antigravity-cli/brain/e97fce9c-463e-4f5b-8f3b-7cb078d54c5d/LiH-GS_Current_DIs.png')
        print("Generated LiH-GS_Current_DIs.png")
except Exception as e:
    print(f"LiH GS Error: {e}")
