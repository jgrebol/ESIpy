import matplotlib.pyplot as plt
import numpy as np

distances = np.arange(3.5, 4.6, 0.1)
e_0 = [-106.819714, -106.824527, -106.847993, -106.887879, -106.941328, -107.005497, -107.077244, -107.153310, -107.231358, -107.309320, -107.385429]
e_1 = [-106.805668, -106.815495, -106.842937, -106.885543, -106.940661, -107.005195, -107.076383, -107.152121, -107.229986, -107.307864, -107.383953]

plt.figure(figsize=(8, 5))
plt.plot(distances, e_0, 'o-', color='#1f77b4', linewidth=2, label='Ground State (Root 0)')
plt.plot(distances, e_1, 's--', color='#ff7f0e', linewidth=2, label='Excited State (Root 1)')
plt.xlabel('Distance R (Å)', fontsize=12)
plt.ylabel('Energy (Hartree)', fontsize=12)
plt.title('LiF SA-CASSCF(6,6) Energies across the Avoided Crossing', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('/home/joan/.gemini/antigravity-cli/brain/e97fce9c-463e-4f5b-8f3b-7cb078d54c5d/LiF_Energy_Smooth_Test.png', dpi=300)
print('Plot generated.')
