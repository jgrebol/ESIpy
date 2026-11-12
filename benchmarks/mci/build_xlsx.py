"""
Copy of MCI_Approximation_Summary.xlsx with the Glynn validation added (the original sheets are left as they are).
  python benchmarks/mci/build_xlsx.py  ->  benchmarks/mci/MCI_Approximation_Summary_Glynn.xlsx
Reads dataset_results.jsonl (run_dataset.py) and large_rings.jsonl (large_rings.py).
"""
import os, json, shutil
import openpyxl

here = os.path.dirname(os.path.abspath(__file__))
SRC = "/home/joan/PycharmProjects/ESIpy/joan/MCIAPROX/MCI_Approximation_Summary.xlsx"
DST = os.path.join(here, "MCI_Approximation_Summary_Glynn.xlsx")
REF = json.load(open("/home/joan/PycharmProjects/ESIpy/joan/MCIAPROX/collected_mci_data_mcialg4.json"))
rows = [json.loads(l) for l in open(os.path.join(here, "dataset_results.jsonl"))]
shutil.copy(SRC, DST)
wb = openpyxl.load_workbook(DST)

# Workbook references are the ESIpy values divided by 2^(n-1); index them by system name
ref = {}
for sec in ("annulenes_approx", "ground_state_approx", "triplet_state_approx"):
    items = REF[sec] if isinstance(REF[sec], list) else REF[sec]["main"] + REF[sec]["separated"]
    for x in items:
        ref[x["name"].split(" ")[0]] = x


def ref_of(r):
    base = os.path.basename(r["system"])
    for k in (base, base + "_s", base.replace("_t", "") + "_t"):
        if k in ref:
            return ref[k]
    return None


def verdict(cell):
    return "" if "match" not in cell else ("PASS" if cell["match"] else "FAIL")


ws = wb.create_sheet("Glynn_Validation")
ws.append(["System", "Ring atoms n", "Wavefunction", "Occupied MOs", "kmax",
           "Exact MCI (Glynn)", "Exact MCI (DFS, same AOMs)", "Exact: Glynn vs DFS", "Exact perms ((n-1)!/2)", "Glynn terms (2^(n-2))", "Exact time Glynn (s)", "Exact time DFS (s)",
           "mcialg=3 MCI (Glynn)", "mcialg=3 perms", "mcialg=3: Glynn vs DFS",
           "mcialg=5 MCI (Glynn)", "mcialg=5 perms", "mcialg=5 MCI (DFS)", "mcialg=5: Glynn vs DFS", "mcialg=5 (% of exact)", "mcialg=5 time Glynn (s)", "mcialg=5 time DFS (s)",
           "Workbook exact x 2^(n-1)", "Glynn / workbook"])
for r in rows:
    if "exact" not in r:
        continue
    e, a3, a5 = r["exact"], r[f"alg3_d{r['kmax']}"], r[f"alg5_d{r.get('kmax_odd', r['kmax'])}"]
    x = ref_of(r)
    wref = x["exact_mci"] * 2 ** (r["n"] - 1) if x and r["n"] == x.get("n", r["n"]) else None
    ws.append([r["system"], r["n"], r["kind"], r["K"], r["kmax"],
               e["value"], e.get("dfs_value"), verdict(e), e["perms"], 2 ** (r["n"] - 2), e["seconds"], e.get("dfs_seconds"),
               a3["value"], a3["perms"], verdict(a3),
               a5["value"], a5["perms"], a5.get("dfs_value"), verdict(a5), 100 * a5["value"] / e["value"] if e["value"] else None,
               a5["seconds"], a5.get("dfs_seconds"), wref, e["value"] / wref if wref else None])

ws = wb.create_sheet("Glynn_Fingerprint")
ws.append(["System", "n", "Exact MCI (Glynn)", "Sum of shells", "Shells of mcialg=4 by distance d = 1, 2, 3, ... (MCI contribution)"])
for r in rows:
    if "alg4_shells" in r:
        s = r["alg4_shells"]
        ws.append([r["system"], r["n"], r["exact"]["value"], s["sum"]] + s["values"])

ws = wb.create_sheet("Glynn_Fullerenes")
ws.append(["Fullerene", "Cage atoms N", "Permutations d=1 (pruned search)", "Workbook permutations", "MCI (mcialg=5, d=1)", "|MCI|^(1/N)", "Workbook |MCI|^(1/N)", "Time (s)", "Cores", "Workbook time (s, 8 cores)"])
full = {x["name"]: x for x in REF["fullerenes"]}
for r in rows:
    if "alg5_d1" in r and "exact" not in r:
        a = r["alg5_d1"]
        key = os.path.basename(r["system"]).lower().replace("-", "_")
        x = full.get(key, {})
        ws.append([r["system"], r["n"], a["perms"], x.get("approx_p"), a["value"], abs(a["value"]) ** (1 / r["n"]), x.get("norm_val"), a["seconds"], r["cores"], x.get("time_sec")])

lr = os.path.join(here, "large_rings.jsonl")
if os.path.exists(lr):
    ws = wb.create_sheet("Glynn_LargeAnnulenes")
    ws.append(["CnHn regular polygon, RHF/STO-3G, meta-Lowdin (built for the scaling test)", "n", "Quantity", "d", "MCI", "Permutations", "Glynn terms", "Time (s)", "Cores", "% of exact"])
    for l in open(lr):
        d = json.loads(l)
        if "value" in d:
            ws.append([f"annulene{d['n']}", d["n"], d["kind"], d.get("d"), d["value"], d.get("perms"), d.get("glynn_terms"), d["seconds"], d.get("cores"),
                       d.get("alg5_percent", d.get("percent_of_exact"))])
wb.save(DST)
print("written", DST, "rows", len(rows))
