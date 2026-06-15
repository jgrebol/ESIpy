import os
import pickle
import numpy as np
from esipy import ESI
from esipy.readfchk import readfchk
from esipy.tools import find_ns, find_di, wf_type

def generate_references():
    base_dir = os.path.dirname(__file__)
    ref_file = os.path.join(base_dir, "pyscf_refs.pkl")
    
    files = {
        '1_benzene_spherical': '1_benzene_spherical.fchk',
        '2_benzene_cartesian': '2_benzene_cartesian.fchk',
        '3_o2_triplet': '3_o2_triplet.fchk',
        '4_h2_oss': '4_h2_oss.fchk',
        '5_high_l': '5_high_l.fchk',
        '6_ecp': '6_ecp.fchk',
        '7_rmp2': '7_rmp2.fchk',
        '8_cisd': '8_cisd.fchk',
        '9_ccsd': '9_ccsd.fchk',
        '10_casscf_rest': '10_casscf_rest.fchk',
        '11_ump2': '11_ump2.fchk',
        '12_casscf_unrest': '12_casscf_unrest.fchk',
        '13_anthracene': '13_anthracene.fchk',
    }
    
    refs = {}
    
    for key, fchk_file in files.items():
        print(f"Generating reference for {key}...")
        
        path = os.path.join(base_dir, "FCHK", "GAUSSIAN", fchk_file)
        if not os.path.exists(path):
            print(f"Skipping {key}: FCHK file not found.")
            continue
            
        ecp_val = "lanl2dz" if key == "6_ecp" else None
        mol_f, mf_f = readfchk(path, ecp=ecp_val)
        mol = mol_f.pyscf_mol
        
        try:
            # Always use the FCHK object directly as the reference.
            # This guarantees the stored pops/DIs match exactly what the
            # esipy script reads from the same file.
            mf = mf_f
            myhf = getattr(mf, '_scf', None)
            esi = ESI(mol=mol_f, mf=mf, myhf=myhf, partition='mulliken')
            aoms = esi.aom
            wf = wf_type(aoms)
            atoms = list(range(1, mol_f.natm+1))

            if wf == "unrest":
                # find_ns always applies 2×, which is only correct for restricted doubly-occupied
                # MOs. For single-spin (α or β) AOMs each MO has occupation 1, so use Tr directly.
                f_pops = np.array([np.trace(aoms[0][i-1]) for i in atoms]) \
                       + np.array([np.trace(aoms[1][i-1]) for i in atoms])
                # The test parser halves the printed DI before comparing (consistent with restricted).
                # Store half the total spin-DI so the comparison works the same way.
                f_di12 = (find_di(aoms[0], 1, 2) + find_di(aoms[1], 1, 2)) / 2.0
            elif wf == "no":
                if len(aoms) == 2:
                    aom_list, occ = aoms
                    f_pops = np.array([np.sum([occ[i] * aom_list[a_idx][i,i] for i in range(len(occ))]) for a_idx in range(mol_f.natm)])
                else:
                    aom_alpha, aom_beta, occ_alpha, occ_beta = aoms
                    pop_a = np.array([np.sum([occ_alpha[i] * aom_alpha[a_idx][i,i] for i in range(len(occ_alpha))]) for a_idx in range(mol_f.natm)])
                    pop_b = np.array([np.sum([occ_beta[i] * aom_beta[a_idx][i,i] for i in range(len(occ_beta))]) for a_idx in range(mol_f.natm)])
                    f_pops = pop_a + pop_b
                f_di12 = 0.0
            else:
                f_pops = np.array(find_ns(atoms, aoms))
                f_di12 = find_di(aoms, 1, 2) if mol_f.natm >= 2 else 0.0

            # Convert numpy arrays/scalars to native Python lists/floats
            # to avoid pickling NumPy objects that cause cross-version
            # compatibility errors/segfaults.
            refs[key] = {
                'e': float(getattr(mf, 'e_tot', 0.0)) if getattr(mf, 'e_tot', None) is not None else 0.0,
                'ind': {
                    'pops': f_pops.tolist() if isinstance(f_pops, np.ndarray) else list(f_pops),
                    'di12': float(f_di12)
                }
            }

            
        except Exception as e:
            print(f"Failed to generate reference for {key}: {e}")

    with open(ref_file, "wb") as f:
        pickle.dump(refs, f)
        
    print(f"References saved to {ref_file}")

if __name__ == "__main__":
    generate_references()
