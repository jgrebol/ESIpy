import os
import glob
import subprocess

fchk_dirs = {
    "bz": "/home/joan/IAOEDU/NEWIAOS/BZ",
    "c4h4": "/home/joan/IAOEDU/C4H4",
    "formamide": "/home/joan/IAOEDU/NEWIAOS/FORMAMIDE"
}

rings = {
    "bz": "1 2 3 4 5 6",
    "c4h4": "1 2 4 3",
    "formamide": "$NORING"
}

def main():
    os.makedirs("robust_benchmark", exist_ok=True)
    
    for stem, d in fchk_dirs.items():
        fchk_files = glob.glob(f"{d}/*.fchk")
        for fchk in fchk_files:
            base = os.path.splitext(os.path.basename(fchk))[0]
            hs_file = f"robust_benchmark/{base}.hs"
            esi_file = f"robust_benchmark/{base}.esi"
            
            with open(hs_file, "w") as f:
                f.write(f"$READFCHK\n{fchk}\n")
                if stem == "formamide":
                    f.write("$NORING\n")
                else:
                    f.write(f"$RING\n{rings[stem]}\n")
                f.write("$PARTITION\nIAO\nALLEFFAO\n")
            
            print(f"Running esipy for {base}...")
            # Use the local esipy script (I'll assume it's in the PATH or I can call it directly)
            # Since I verified it's in /home/joan/.local/bin/esipy or I can call it via python
            try:
                # We want to use the current development version, so PYTHONPATH should point to local repo
                env = os.environ.copy()
                env["PYTHONPATH"] = "/home/joan/PycharmProjects/ESIpy"
                
                with open(esi_file, "w") as out:
                    subprocess.run(["python3", "-m", "esipy.cli", hs_file], 
                                   stdout=out, stderr=subprocess.STDOUT, env=env)
            except Exception as e:
                print(f"Error running {base}: {e}")

if __name__ == "__main__":
    main()
