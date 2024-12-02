from distutils.core import setup

setup(name="pyESI",
        version="1.0",
        description="Calculation of electronic aromaticity indicators",
        author="Joan Grebol, Eduard Matito, Pedro Salvador",
        url="https://github.com/jgrebol/ESIpy",
        packages=["esipy"],
      requires=['numpy', 'pyscf', 'scipy'],
        )
