import os
import iodata
from iodata.api import _select_format_module


def convert_to_fchk(filename, out_fchk=None, fmt=None):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File '{filename}' does not exist.")

    if filename.lower().endswith(('.fchk', '.fch')) and out_fchk is None:
        return filename

    if fmt is None:
        try:
            mod = _select_format_module(filename, "load_one", None)
            fmt_name = mod.__name__.split(".")[-1].upper()
        except Exception:
            fmt_name = "UNKNOWN"
    else:
        fmt_name = fmt.upper()

    print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")
    print(f" | File conversion from {fmt_name} to FCHK using IOData")
    print(" | Reference: T. Verstraelen et al., J. Comput. Chem. 2021, 42(6), 458-464.")
    print(" | https://github.com/theochem/iodata")
    print(" -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ")

    data = iodata.load_one(filename, fmt=fmt)
    if data.mo is None and data.obasis is None:
        raise ValueError(f"File '{filename}' does not contain wavefunction or orbital data readable by IOData.")

    if out_fchk is None:
        base, _ = os.path.splitext(filename)
        out_fchk = f"{base}.esipy.fchk"

    iodata.dump_one(data, out_fchk, fmt="fchk")
    return out_fchk


def parse(filename, ecp=None, out_fchk=None, fmt=None):
    from esipy.readfchk import readfchk
    fchk_path = convert_to_fchk(filename, out_fchk=out_fchk, fmt=fmt)
    return readfchk(fchk_path, ecp=ecp)


parse_and_read = parse
