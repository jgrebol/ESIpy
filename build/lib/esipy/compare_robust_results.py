import os
import glob
import csv
import re

REFERENCE_CSV = '/home/joan/IAOEDU/REFERENCE/recent_actual_values.csv'

def load_reference():
    ref_data = {}
    if not os.path.exists(REFERENCE_CSV): return ref_data
    with open(REFERENCE_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['section'].lower() == 'charge' and row['method'].upper() == 'IAO':
                # Store keys in lowercase
                key = (row['stem'].lower(), row['item'].lower())
                ref_data[key] = float(row['method_value'])
    return ref_data

def parse_esi(esi_file):
    results = {} # method -> {item -> pop}
    current_method = None
    with open(esi_file, "r") as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i]
        if "Atomic partition:" in line:
            current_method = line.split(":")[1].strip().upper()
            if current_method not in results:
                results[current_method] = {}
        
        if "| Atom    N(Sij)" in line:
            i += 2 # skip header and separator
            while i < len(lines) and "|" in lines[i] and "TOT:" not in lines[i]:
                parts = lines[i].split("|")
                if len(parts) >= 3:
                    atom_info = parts[1].strip().split()
                    if len(atom_info) >= 2:
                        sym = atom_info[0]
                        idx = atom_info[1]
                        pop = float(parts[2].strip())
                        item = f"{sym}{idx}".lower()
                        results[current_method][item] = pop
                i += 1
        i += 1
    return results

def main():
    ref_data = load_reference()
    print("DEBUG: first 5 ref_data keys:", list(ref_data.keys())[:5])
    esi_files = glob.glob("robust_benchmark/*.esi")
    
    # Symbols to Z mapping
    Z_map = {"C": 6, "H": 1, "O": 8, "N": 7}

    print(f"{'Stem':<30} {'Item':<6} {'Calc Charge':>12} {'Ref Charge':>12} {'Diff':>10}")
    print("-" * 75)
    
    all_diffs = []

    for esi in sorted(esi_files):
        stem = os.path.splitext(os.path.basename(esi))[0].lower()
        data = parse_esi(esi)
        
        if "IAO" in data:
            iao_data = data["IAO"]
            for item, pop in iao_data.items():
                sym = re.match(r"([a-zA-Z]+)", item).group(1).upper()
                Z = Z_map.get(sym, 0)
                calc_charge = Z - pop
                key = (stem, item)
                ref_charge = ref_data.get(key, None)
                
                if ref_charge is not None:
                    diff = calc_charge - ref_charge
                    all_diffs.append(abs(diff))
                    print(f"{stem:<30} {item:<6} {calc_charge:12.6f} {ref_charge:12.6f} {diff:10.2e}")
                else:
                    # Debug print for first few failures
                    if len(all_diffs) < 5:
                        pass
                        # print(f"DEBUG: Failed lookup for {key}")

    if all_diffs:
        print("-" * 75)
        print(f"Max Absolute Difference:  {max(all_diffs):.2e}")
        print(f"Mean Absolute Difference: {sum(all_diffs)/len(all_diffs):.2e}")

if __name__ == "__main__":
    main()
