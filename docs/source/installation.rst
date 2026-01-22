============
Installation
============


The recommended method of installation is to use the Python package manager |pip|_:

.. code-block:: console

    $ pip install weas-widget

This will install the latest stable version that was released to PyPI.

To install the package from source, first clone the repository and then install using |pip|_:

.. code-block:: console

    $ git clone https://github.com/superstar54/weas-widget
    $ pip install -e weas-widget

The ``-e`` flag will install the package in editable mode, meaning that changes to the source code will be automatically picked up.

Optional features can be installed with extras. For example, the Fermi surface feature
requires ``seekpath``:

.. code-block:: console

    $ pip install "weas-widget[fermi-surface]"



.. |pip| replace:: ``pip``
.. _pip: https://pip.pypa.io/en/stable/
