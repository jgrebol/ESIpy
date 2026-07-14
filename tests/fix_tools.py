import numpy as np

with open('../esipy/tools.py', 'r') as f:
    lines = f.readlines()

start_gaussian = -1
for i, line in enumerate(lines):
    if 'def permute_aos_rows(mat, mole2):' in line:
        start_gaussian = i
        break

new_lines = lines[:start_gaussian]

combined_code = """def permute_aos_rows(mat, mole2):
    mol = mole2.pyscf_mol
    is_cart = bool(getattr(mol, 'cart', False))
    pi = np.pi
    p1 = 2.0 * np.sqrt(pi / 15.0)
    p2 = 2.0 * np.sqrt(pi / 5.0)
    p3 = 2.0 * np.sqrt(pi / 7.0)
    p4 = 2.0 * np.sqrt(pi / 35.0)
    p5 = 2.0 * np.sqrt(pi / 105.0)
    p6 = (2.0 / 3.0) * np.sqrt(pi)
    p7 = (2.0 / 3.0) * np.sqrt(pi / 7.0)
    p8 = (2.0 / 3.0) * np.sqrt(pi / 35.0)
    p9 = 2.0 * np.sqrt(pi / 11.0)
    p10 = (2.0 / 3.0) * np.sqrt(pi / 11.0)
    p11 = 2.0 * np.sqrt(pi / 231.0)
    p12 = (2.0 / 3.0) * np.sqrt(pi / 77.0)
    p13 = 2.0 * np.sqrt(pi / 1155.0)
    SDIAG = {
        2: [p2, p1, p1, p2, p1, p2],
        3: [p3, p4, p4, p4, p5, p4, p3, p4, p4, p3],
        4: [p6, p7, p7, p5, p8, p5, p7, p8, p8, p7, p6, p7, p5, p7, p6],
        5: [p9, p10, p10, p11, p12, p11, p11, p13, p13, p11, p10, p12, p13, p12, p10, p9, p10, p11, p11, p10, p9]
    }
    MAPS = {
        2: [0, 3, 4, 1, 5, 2],
        3: [0, 4, 5, 3, 9, 6, 1, 8, 7, 2],
        4: [0, 4, 5, 3, 14, 6, 11, 13, 12, 9, 1, 8, 7, 10, 2],
        -2: [4, 2, 0, 1, 3],
        -3: [6, 4, 2, 0, 1, 3, 5],
        -4: [8, 6, 4, 2, 0, 1, 3, 5, 7],
    }
    atom_map = np.asarray(mole2.fchk_basis_arrays['iatsh']) - 1
    shell_types = np.asarray(mole2.fchk_basis_arrays['mssh'])
    registry = {}
    cursor = 0
    for iat, st in zip(atom_map, shell_types):
        if st == -1:
            key_s = (int(iat), 0); registry.setdefault(key_s, []).append({'start': cursor, 'count': 1})
            key_p = (int(iat), 1); registry.setdefault(key_p, []).append({'start': cursor+1, 'count': 3})
            cursor += 4
        else:
            l = abs(st)
            n = (l+1)*(l+2)//2 if st>=0 and l>1 else (3 if l==1 else (2*l+1 if st<0 else 1))
            key = (int(iat), l); registry.setdefault(key, []).append({'start': cursor, 'count': n})
            cursor += n
    idx_list, scale_list, usage = [], [], {}
    for b in range(mol.nbas):
        iat, l, nctr = int(mol.bas_atom(b)), int(mol.bas_angular(b)), int(mol.bas_nctr(b))
        for _ in range(nctr):
            key = (iat, l); count = usage.get(key, 0); target = registry[key][count]; usage[key] = count + 1
            if l <= 1:
                idx_list.extend(range(target['start'], target['start'] + target['count']))
                scale_list.extend([1.0] * target['count'])
            else:
                lookup = l if is_cart else -l
                order = MAPS.get(lookup, list(range(target['count'])))
                idx_list.extend([target['start'] + i for i in order])
                scale_list.extend(SDIAG.get(lookup, [1.0] * target['count']))
    p, s = np.array(idx_list), np.array(scale_list)
    if mat.ndim == 2: return mat[p] * s[:, None] if mat.shape[0] == len(p) else mat[:, p] * s[None, :]
    if mat.ndim == 3: return mat[:, p, :] * s[None, :, None] if mat.shape[1] == len(p) else mat[:, :, p] * s[None, None, :]
    return mat

def permute_aos_rows_qchem(mat, mole2):
    return permute_aos_rows(mat, mole2)
"""

with open('../esipy/tools.py', 'w') as f:
    f.writelines(new_lines)
    f.write(combined_code)
