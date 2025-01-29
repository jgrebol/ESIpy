from setuptools import setup

setup(
    name="ESIpython",
    version="1.0",
    description="Calculation of electronic aromaticity indicators",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Joan Grebol, Eduard Matito, Pedro Salvador",
    url="https://github.com/jgrebol/ESIpy",
    packages=["esipy"],
    install_requires=[
        'numpy==2.1.3',
        'pyscf==2.7.0',
        'scipy==1.14.1'
    ],
)
