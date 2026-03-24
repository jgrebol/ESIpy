from setuptools import setup

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
        "scripts/esipy"
    ],
    install_requires=[
        "numpy",
        "pyscf<=2.8.0",
        "sphinx",
        "urllib3>=2.0.0",
        "h5py",
    ],
    python_requires=">=3.7",
)

