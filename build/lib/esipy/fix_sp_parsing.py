import re

def fix():
    with open('readfchk.py', 'r') as f:
        content = f.read()

    # Improved make_basis for SP shells
    new_sp_logic = """        if l_raw == -1:
            if c2 is not None:
                # S and P in separate arrays (Gaussian format)
                for _ in range(n_prim):
                    primitives.append((expsh[exp_idx], c1[coeff_idx], c2[coeff_idx]))
                    exp_idx += 1; coeff_idx += 1
                done_shells[sym].append({"l": -1, "primitives": primitives})
            else:
                # S and P interleaved or concatenated in c1
                # Gaussian format in c1: [s1, s2, ..., sn, p1, p2, ..., pn]
                for i in range(n_prim):
                    primitives.append((expsh[exp_idx + i], c1[coeff_idx + i], c1[coeff_idx + n_prim + i]))
                exp_idx += n_prim; coeff_idx += 2 * n_prim
                done_shells[sym].append({"l": -1, "primitives": primitives})
"""

    pattern_sp = re.compile(r"if l_raw == -1:.*?done_shells\[sym\]\.append\(\{\"l\": -1, \"primitives\": primitives\}\)", re.DOTALL)
    content = pattern_sp.sub(new_sp_logic, content)
    
    with open('readfchk.py', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    fix()
