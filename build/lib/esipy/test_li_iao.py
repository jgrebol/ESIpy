import numpy as np
from pyscf import gto, scf
from esipy import make_aoms

def test_li_counts():
    mol = gto.M(atom='Li 0 0 0', basis='ccpvdz', spin=1, verbose=0)
    mf = scf.UHF(mol)
    mf.kernel()
    
    partitions = ['iao', 'iao2', 'peiao', 'iao-effao']
    
    print("--- Lithium Orbital Counts ---")
    for p in partitions:
        try:
            aoms = make_aoms(mol, mf, partition=p)
            # AOMs for UHF/CASSCF are [alpha, beta]
            # Dimensions should match the IAO space size
            n_ref = aoms[0][0].shape[0]
            print(f"Partition: {p:<10} | IAO Space Size: {n_ref}")
        except Exception as e:
            print(f"Partition: {p:<10} | FAILED: {e}")

if __name__ == "__main__":
    test_li_counts()
