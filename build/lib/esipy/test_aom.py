import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from esipy.readfchk import readfchk
from esipy.make_aoms import make_aoms
import numpy as np

path = '../joan/LiH/GS/lih_1.0.fchk'
mol, mf, myhf = readfchk(path)

print("Running make_aoms with partition='mulliken'")
aom_mulliken = make_aoms(mol, mf, 'mulliken', myhf=myhf)
# aom_mulliken for NO is [aom_list, occ_matrix]
aoms, occ = aom_mulliken
print("Mulliken AOM shape:", aoms[0].shape)
print("Occ shape:", occ.shape)
pop0 = np.trace(occ @ aoms[0])
pop1 = np.trace(occ @ aoms[1])
print(f"Mulliken Populations: {pop0:.4f}, {pop1:.4f}, Total: {pop0+pop1:.4f}")

print("\nRunning make_aoms with partition='iao'")
aom_iao = make_aoms(mol, mf, 'iao', myhf=myhf)
aoms_iao, occ_iao = aom_iao
pop0_iao = np.trace(occ_iao @ aoms_iao[0])
pop1_iao = np.trace(occ_iao @ aoms_iao[1])
print(f"IAO Populations: {pop0_iao:.4f}, {pop1_iao:.4f}, Total: {pop0_iao+pop1_iao:.4f}")
