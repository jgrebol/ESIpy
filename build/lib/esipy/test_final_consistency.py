
import numpy as np
from pyscf import gto, scf, mp
from esipy import ESI

def test_aromaticity():
    # 1. Restricted (Benzene)
    mol = gto.M(atom='C 0 0 0; C 1.39 0 0; C 2.085 1.2037 0; C 1.39 2.4074 0; C 0 2.4074 0; C -0.695 1.2037 0',
                basis='sto-3g')
    mf = scf.RHF(mol).run()
    esi = ESI(mol=mol, mf=mf, partition='mulliken')
    print(f"\n[REST] Ring: {esi.rings[0]}")
    print(f" Iring: {esi.indicators[0].iring:.6f}")
    print(f" PDI:   {esi.indicators[0].pdi:.6f}")
    print(f" FLU:   {esi.indicators[0].flu:.6f}")

    # 2. Unrestricted (O2 triplet)
    mol_o2 = gto.M(atom='O 0 0 0; O 0 0 1.208', basis='sto-3g', spin=2)
    mf_o2 = scf.UHF(mol_o2).run()
    esi_o2 = ESI(mol=mol_o2, mf=mf_o2, partition='mulliken', rings=[[1, 2]])
    print(f"\n[UNREST] Ring: {esi_o2.rings[0]}")
    print(f" Iring: {esi_o2.indicators[0].iring:.6f}")
    flu = esi_o2.indicators[0].flu
    print(f" FLU:   {flu if flu is None else f'{flu:.6f}'}")

    # 3. Natural Orbitals (MP2 Benzene)
    mf_mp2 = mp.MP2(mf).run()
    esi_no = ESI(mol=mol, mf=mf_mp2, partition='mulliken')
    print(f"\n[NO] Ring: {esi_no.rings[0]}")
    print(f" Iring: {esi_no.indicators[0].iring:.6f}")
    print(f" PDI:   {esi_no.indicators[0].pdi:.6f}")
    print(f" FLU:   {esi_no.indicators[0].flu:.6f}")

if __name__ == "__main__":
    test_aromaticity()
