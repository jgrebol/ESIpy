import os
import numpy as np
import matplotlib.pyplot as plt
from esipy.readfchk import readfchk

distances = np.arange(1.6, 7.1, 0.1)
energies = []
valid_dist = []

for r in distances:
    fchk_file = f"/home/joan/PycharmProjects/ESIpy/joan/HARPOON/LiF/lif_{r:.1f}.fchk"
    if os.path.exists(fchk_file):
        try:
            _, mf = readfchk(fchk_file)
            energies.append(mf.e_tot)
            valid_dist.append(r)
        except Exception as e:
            pass

plt.figure(figsize=(8, 5))
plt.plot(valid_dist, energies, 'o-', color='#d62728', linewidth=2, label='FCHK Energy')
plt.xlabel('Distance R (Å)', fontsize=12)
plt.ylabel('Energy (Hartree)', fontsize=12)
plt.title('LiF FCHK Total Energy Profile', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('/home/joan/.gemini/antigravity-cli/brain/e97fce9c-463e-4f5b-8f3b-7cb078d54c5d/LiF_FCHK_Energies.png', dpi=300)
print('FCHK Energy plot generated.')
