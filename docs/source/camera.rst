===================
Camera
===================

Setting
=============

Camera has three settings:

- zoom: the zoom level of the camera
- position: the position of the camera
- look_at: the point the camera is looking at

.. code-block:: python

    viewer.camera.zoom = 2
    viewer.camera.position = [0, 0, 100]
    viewer.camera.look_at = [0, 0, 0]


If you want to set the `direction` of the camera, you can use the following code:

.. code-block:: python

    # the look_at is the center of the atoms (or the center of the bounding box in case of no atoms)
    # the distance is the distance between the camera and the look_at point
    viewer.camera.setting = {"direction": [0, 5, 1], "distance": 50, "zoom": 2}


Camera Type
=============

For the moment, only orthographic camera is supported.
