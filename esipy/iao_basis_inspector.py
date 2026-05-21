import numpy as np
from pyscf import gto, scf
import esipy
from esipy.iao import reference_mol, get_num_minbas_per_l
from esipy.make_aoms import format_partition

def inspect_partition_basis(symbol='Li', basis='cc-pVTZ', partitions=None):
    """
    Analyzes the reference basis size and composition for a given atom and list of partitions.
    """
    if partitions is None:
        partitions = ['iao', 'iao2', 'iao-effao-net', 'peiao', 'fpiao(1.5)', 'dfpiao(0.5)']

    # Create a dummy molecule for the atom
    mol = gto.M(atom=f'{symbol} 0 0 0', basis=basis, spin=1, verbose=0)
    
    print(f"============================================================")
    print(f" IAO BASIS INSPECTOR: {symbol} (Working Basis: {basis})")
    print(f"============================================================")
    print(f"{'Partition':<20} | {'Total Orbs':<10} | {'Reference Shells (l: count)'}")
    print("-" * 60)

    for p_label in partitions:
        try:
            # 1. Parse partition to get defaults (logic from make_aoms)
            # We'll mimic the default resolution here
            p_base = p_label.split('(')[0].lower()
            
            if p_base == "iao":
                iaoref = 'minao'
                is_pol = False
            elif p_base in ["iao2", "iao-effao", "iao-autosad", "peiao", "dpeiao", "fpiao", "dfpiao"] or p_base.startswith("iao-effao"):
                iaoref = 'valence' # which resolves to sto-3g size
                is_pol = "p" in p_base # peiao, fpiao, dpeiao, dfpiao
            else:
                iaoref = 'minao'
                is_pol = False

            # 2. Get the shell counts using the new logic in iao.py
            shells = get_num_minbas_per_l(symbol, polarized=is_pol, source_basis=iaoref)
            
            # 3. Calculate total orbitals
            total_orbs = sum(count * (2*l + 1) for l, count in shells.items())
            
            # Format shells string
            shell_str = ", ".join([f"{l}:{c}" for l, c in sorted(shells.items())])
            
            print(f"{p_label:<20} | {total_orbs:<10} | {shell_str}")

        except Exception as e:
            print(f"{p_label:<20} | {'ERROR':<10} | {e}")

    print("-" * 60)
    print("Definitions:")
    print("  MINAO (iao): Core + occupied valence shells.")
    print("  VALENCE (iao2, effao): Core + all valence shells (STO-3G size).")
    print("  POLARIZED (peiao, fpiao): Valence + next angular momentum layer.")
    print("============================================================")

if __name__ == "__main__":
    import sys
    import os
    
    # Check if we are using the local version
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    target_atom = sys.argv[1] if len(sys.argv) > 1 else 'Li'
    inspect_partition_basis(symbol=target_atom)
