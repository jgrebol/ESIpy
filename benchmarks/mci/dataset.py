"""
AOMs of the MCIAPROX dataset (joan/MCIAPROX/ALLDATASET/GEOMS) from the .fchk files, IAO partition, cached on disk.
  load("ANNULENES/c14h14_d14h") -> {"aom", "connec", "rings", "kind"}
One system at a time; run under `ulimit -v` (the fchk are aug-cc-pVDZ).
"""
import os, io, pickle, contextlib

GEOMS = "/home/joan/PycharmProjects/ESIpy/joan/MCIAPROX/ALLDATASET/GEOMS"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "aoms")


def read_rings(hs):
    lines = [l.strip() for l in open(hs)]
    rings, i = [], lines.index("$RING") + 1 if "$RING" in lines else len(lines)
    while i < len(lines) and lines[i] and not lines[i].startswith("$"):
        rings.append([int(x) for x in lines[i].replace(",", " ").split()])
        i += 1
    return rings


def load(rel, partition="iao"):
    out = os.path.join(CACHE, rel.replace("/", "__") + ".pkl")
    if os.path.exists(out):
        return pickle.load(open(out, "rb"))
    from esipy import ESI
    from esipy.readfchk import readfchk
    from esipy.tools import wf_type
    base = os.path.join(GEOMS, rel)
    rings = read_rings(base + ".hs")
    with contextlib.redirect_stdout(io.StringIO()):
        cwd = os.getcwd()
        os.chdir(os.path.dirname(base))
        try:
            mol, mf = readfchk(os.path.basename(base) + ".fchk")
            esi = ESI(mol=mol, mf=mf, partition=partition, rings=rings)
            data = {"aom": esi.aom, "connec": esi.connec, "rings": rings, "kind": wf_type(esi.aom)}
        finally:
            os.chdir(cwd)
    os.makedirs(CACHE, exist_ok=True)
    pickle.dump(data, open(out, "wb"))
    return data


if __name__ == "__main__":
    import sys, time, resource
    for rel in sys.argv[1:]:
        t = time.time()
        d = load(rel)
        a = d["aom"][0] if d["kind"] == "unrest" else d["aom"]
        print(rel, d["kind"], "rings", [len(r) for r in d["rings"]], "K", a[0].shape[0], f"{time.time() - t:.1f}s",
              f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3:.0f} MB", flush=True)
