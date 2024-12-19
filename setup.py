from distutils.core import setup

setup(name="ESIpython",
      version="1.0",
      description="Calculation of electronic aromaticity indicators",
      author="Joan Grebol, Eduard Matito, Pedro Salvador",
      url="https://github.com/jgrebol/ESIpy",
      packages=["esipy"],
      requires=['numpy=2.1.3', 'pyscf=2.7.0', 'scipy=1.14.1'],
      )
