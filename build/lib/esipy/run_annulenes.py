import os
import subprocess
import glob

def main():
    base_data_path = "/home/joan/IAOEDU/ANNULENES"
    output_path = "/home/joan/IAOEDU/OUTPUTS"
    esipy_script = "/home/joan/PycharmProjects/ESIpy/scripts/esipy"
    python_path = "/home/joan/PycharmProjects/ESIpy"
    
    fchks = glob.glob(os.path.join(base_data_path, "*.fchk"))
    
    for fchk in fchks:
        basename = os.path.splitext(os.path.basename(fchk))[0]
        hs_file = os.path.join(output_path, f"{basename}_annulene.hs")
        esi_file = os.path.join(output_path, f"{basename}_annulene.esi")
        
        with open(hs_file, "w") as f:
            f.write(f"$READFCHK\n{fchk}\n")
            f.write("$FINDRINGS\n$MINLEN\n4\n$MAXLEN\n18\n")
            f.write("$IAOMIX\n0.0 0.25 0.5 0.75 1.0\n")
            f.write("$PARTITION\nALLWIP\n")
        
        print(f"Running calculation for {basename}...")
        env = os.environ.copy()
        env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":" + python_path
        
        with open(esi_file, "w") as out:
            try:
                subprocess.run(["python3", esipy_script, hs_file], 
                               env=env, stdout=out, stderr=subprocess.STDOUT, text=True)
            except Exception as e:
                print(f"Error running {basename}: {e}")

if __name__ == "__main__":
    main()
