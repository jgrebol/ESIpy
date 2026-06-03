import os

def fix():
    with open('tools.py', 'r') as f:
        lines = f.readlines()

    new_lines = []
    skip = False
    for line in lines:
        if "if 'ccsd' in str(mf.__class__).lower()" in line:
            new_lines.append("    rdm1 = mf.make_rdm1(ao_repr=True)\n")
            skip = True
        elif skip and "rdm1_arr = np.asarray(rdm1)" in line:
            new_lines.append(line)
            skip = False
        elif not skip:
            new_lines.append(line)

    with open('tools.py', 'w') as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    fix()
