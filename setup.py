from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import shutil
import subprocess
from pathlib import Path


# --- detect compiler (unchanged) ---
def _detect_compiler():
    cc = shutil.which('gcc') or shutil.which('cc') or shutil.which('clang')
    if not cc:
        return None
    try:
        out = subprocess.check_output([cc, '--version'], stderr=subprocess.STDOUT, timeout=2)
        s = out.decode('utf-8', errors='ignore').lower()
        if 'gcc' in s or 'gnu' in s:
            return 'gcc'
        if 'clang' in s:
            return 'clang'
    except Exception:
        return None
    return None


compiler = _detect_compiler()

extra_compile_args = ['-O3', '-fPIC']
extra_link_args = []
if compiler == 'gcc':
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')


# --- Custom build_ext command to build a shared library instead of a Python extension ---
class BuildCLibrary(build_ext):
    """Custom build_ext that compiles mci.c into libmci.so (not a Python module)."""
    def build_extension(self, ext):
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        output_dir = Path(self.get_ext_fullpath(ext.name)).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        src = Path(ext.sources[0])
        lib_name = "libmci.so"  # You could also name it .dylib on macOS
        out_path = output_dir / lib_name

        cc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
        if not cc:
            raise RuntimeError("No C compiler found")

        cmd = [cc, "-shared", "-fPIC", "-O3", "-o", str(out_path), str(src)] + extra_compile_args + extra_link_args
        print("Compiling:", " ".join(cmd))
        subprocess.check_call(cmd)

        print(f"Built shared library: {out_path}")


# --- Define a dummy Extension to trigger our build command ---
mci_ext = Extension("esipy._mci_placeholder", sources=["esipy/mci.c"])


# --- Setup configuration ---
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="esipython",
    version="1.0.6",
    description="Calculation of electronic aromaticity indicators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Joan Grebol, Eduard Matito, Pedro Salvador",
    url="https://github.com/jgrebol/ESIpy",
    packages=["esipy"],
    scripts=[
        "scripts/int2esi",
        "scripts/aom2esi",
        "scripts/fchk2esi",
        "scripts/input2esi",
    ],
    install_requires=[
        "numpy<=1.23.3",
        "pyscf<=2.4",
        "sphinx",
    ],
    python_requires=">=3.7, <4",
    cmdclass={"build_ext": BuildCLibrary},  # <-- build libmci.so automatically
    ext_modules=[mci_ext],
    package_data={"esipy": ["libmci.so", "*.c"]},
    include_package_data=True,
)

