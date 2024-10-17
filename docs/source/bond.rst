Bond
===============
Use can control the bond using `avr.bond.settings`. For example, we delete the bond between Ca and Ti, and Ti and Ca.

.. code-block:: python

    # delete the bond between Ca and Ti
    del viewer1.avr.bond.settings['[Ti, Ca]']
    # change the bond color between Ti and O
    viewer1.avr.bond.settings['[Ti, O]'].update({"color": "red"})
    # change the maximum bond length between Ti and O
    viewer1.avr.bond.settings['[Ti, O]']["max"] = 3.0
