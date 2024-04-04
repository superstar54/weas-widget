===================
Camera
===================

The camera can be configured in two ways:

Direct Camera Settings
=======================

The camera has three direct settings:

- **zoom**: Controls the zoom level of the camera.
- **position**: Specifies the camera's position in 3D space.
- **look_at**: Determines the point in space the camera is oriented towards.

Example usage:

.. code-block:: python

    viewer.camera.zoom = 2
    viewer.camera.position = [0, 0, 100]
    viewer.camera.look_at = [0, 0, 0]

Viewpoint-Centric Settings
===========================
This approach is useful for orienting the camera towards a subject, like the center of atoms or a bounding box, based on direction and distance from the subject.
This method automatically calculates the appropriate `position` and `look_at`` values.

.. code-block:: python

    # Direction is relative to the center of atoms or bounding box.
    # Distance specifies how far the camera is from the look_at point.
    viewer.camera.setting = {"direction": [0, 5, 1], "distance": 50, "zoom": 2}

Camera Type
=============

For the moment, only orthographic camera is supported.
