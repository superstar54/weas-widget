Cell
===============
Use can control the cell using `avr.cell.settings`.


.. code-block:: python

    # delete the cell between Ca and Ti
    viewer.avr.cell.settings['cellLineWidth'] = 1
    viewer.avr.cell.settings['cellColor'] = 'red'
    # hide the crystal axes
    viewer.avr.cell.settings['showAxes'] = False
