import re

def apply_fix():
    with open('readfchk.py', 'r') as f:
        content = f.read()

    # The issue is that the script replace logic removed 'from esipy.tools import permute_aos_rows'
    # but the one inside the string was not properly indented or lost.

    # Let's just do a clean WRITE of the MeanField2 class or the whole file if needed.
    # Actually, let's just fix the import.
    
    content = content.replace("def standardize_mat(mat):", "def standardize_mat(mat):\n            from esipy.tools import permute_aos_rows")
    
    with open('readfchk.py', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    apply_fix()
