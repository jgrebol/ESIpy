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
        "scripts/esipy",
    ],
    install_requires=[
        "pyscf<=2.4",
        "sphinx",
        "urllib3>=2.6.3",
        "h5py",
    ],
    requires_python=">=3.7, <3.12",
)

