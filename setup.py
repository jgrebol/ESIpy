from setuptools import setup

# Read the long description from README.md
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
        "scripts/esipy",
    ],
    install_requires=[
        "numpy<=1.23.3",
        "pyscf<=2.4",
        "sphinx",
        "numba<=0.56.4",
    ],
    requires_python=">=3.7, <3.12",
)

