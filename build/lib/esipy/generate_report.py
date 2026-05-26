import csv
import os

def load_reference(ref_path):
    data = {}
    with open(ref_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize stem and method for matching
            stem = row['stem'].lower()
            method = row['method'].upper()
            section = row['section'].lower()
            item = row['item'].lower()
            try:
                val = float(row['method_value'])
            except:
                continue
            
            # Map reference method names to our benchmark method names
            # FPIAO(1.00) -> FPIAO(1.0)
            if 'FPIAO(' in method and 'XIAO' not in method:
                try:
                    v = float(method.split('(')[1].split(')')[0])
                    method = f'FPIAO({v:.1f})'
                except: pass
            
            # XIAO_FPIAO1(0.30) -> XIAO_DFPIAO(0.3)
            if 'XIAO_FPIAO1(' in method:
                try:
                    v = float(method.split('(')[1].split(')')[0])
                    method = f'XIAO_DFPIAO({v:.1f})'
                except: pass

            key = (stem, method, section, item)
            data[key] = val
    return data

def load_benchmarks(bench_path):
    data = {}
    if not os.path.exists(bench_path):
        print(f"DEBUG: {bench_path} not found")
        return data
    with open(bench_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('stem'): continue
            stem = row['stem'].lower()
            method = row['method'].upper()
            section = row['section'].lower()
            item = row['item'].lower()
            try:
                val = float(row['value'])
            except:
                continue
            key = (stem, method, section, item)
            data[key] = val
    return data

def generate_report(ref_data, bench_data):
    # Focus on ring indices first
    stems = sorted(list(set(k[0] for k in bench_data.keys())))
    print(f"DEBUG: Found {len(stems)} unique stems in bench_data")
    
    with open('comparison_report.tex', 'w') as f:
        f.write(r"\documentclass{article}" + "\n")
        f.write(r"\usepackage[utf8]{inputenc}" + "\n")
        f.write(r"\usepackage{booktabs}" + "\n")
        f.write(r"\usepackage{geometry}" + "\n")
        f.write(r"\geometry{a4paper, margin=1in}" + "\n")
        f.write(r"\title{ESIpy IAO Benchmark Comparison}" + "\n")
        f.write(r"\begin{document}" + "\n")
        f.write(r"\maketitle" + "\n")
        
        for stem in stems:
            stem_keys = [k for k in bench_data.keys() if k[0] == stem and k[2] == 'ring']
            if not stem_keys:
                continue
            
            print(f"DEBUG: Writing section for {stem} ({len(stem_keys)} ring entries)")
            f.write(r"\section{Molecule: " + stem.replace('_', r'\_') + "}\n")
            
            # Ring Indices Table
            f.write(r"\subsection{Ring Indices}" + "\n")
            f.write(r"\begin{tabular}{lcccc}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Method & Indicator & Reference & Benchmark & Diff \\" + "\n")
            f.write(r"\midrule" + "\n")
            
            # Sort by method then item (Iring/MCI)
            stem_keys.sort(key=lambda x: (x[1], x[3]))
            
            for k in stem_keys:
                ref_val = ref_data.get(k, None)
                bench_val = bench_data[k]
                diff = bench_val - ref_val if ref_val is not None else 0.0
                ref_str = f"{ref_val:.6f}" if ref_val is not None else "N/A"
                diff_str = f"{diff:.6f}" if ref_val is not None else "N/A"
                method_name = k[1].replace('_', r'\_')
                
                f.write(f"{method_name} & {k[3]} & {ref_str} & {bench_val:.6f} & {diff_str} \\\\\n")
            
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n\n")
            
        f.write(r"\end{document}" + "\n")

def main():
    ref_path = '/home/joan/IAOEDU/REFERENCE/recent_actual_values.csv'
    bench_path = 'benchmark_results.csv'
    
    ref_data = load_reference(ref_path)
    bench_data = load_benchmarks(bench_path)
    print(f"DEBUG: Loaded {len(ref_data)} ref entries and {len(bench_data)} bench entries")
    
    generate_report(ref_data, bench_data)
    print("Generated comparison_report.tex")

if __name__ == "__main__":
    main()
