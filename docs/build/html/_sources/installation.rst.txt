Installation
************

ESIpy can be installed through different methods. In all cases, it is recommended to create a conda environment:

.. code-block:: bash
    conda create --name esipy python=3.8
    conda activate esipy

Installing via pip
------------------

ESIpy can be installed via pip:

.. code-block:: bash

    pip install esipython

To install PySCF, follow the official guidelines from `PySCF's installation guide <https://pyscf.org/install.html>`_:

.. code-block:: bash

    pip install --prefer-binary pyscf

Installing from source
----------------------

To install ESIpy from source, follow these steps:

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/jgrebol/ESIpy.git
    cd ESIpy

2. Install the package:

.. code-block:: bash

    pip install .