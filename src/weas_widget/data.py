from .base_class import Collection


def deserialize_threejs(json_data):
    """Deserialize a JSON object (from three.js format) into a Object3D instance."""
    geometries = {g["uuid"]: Geometry(**g) for g in json_data.get("geometries", [])}
    materials = {m["uuid"]: Material(**m) for m in json_data.get("materials", [])}

    obj_data = json_data.get("object")
    object3d = Object3D(
        uuid=obj_data["uuid"],
        type=obj_data["type"],
        geometry=geometries.get(obj_data.get("geometry")),
        material=materials.get(obj_data.get("material")),
        **{
            k: v
            for k, v in obj_data.items()
            if k not in ["uuid", "type", "geometry", "material"]
        },
    )
    return object3d


class Data:
    def __init__(self, _widget):
        self._widget = _widget
        self.objects = Collection(_widget)
        self.geometries = Collection(_widget)
        self.materials = Collection(_widget)
        # observe python_task
        self._widget.observe(self._update_object, "python_task")

    def _update_object(self, change):
        if not change["new"] or change["new"]["catalog"] != "object":
            return
        action = change["new"]["action"]
        object = deserialize_threejs(change["new"]["data"])
        if action == "add":
            self.objects.add(object.uuid, object)
            # self.geometries.add(object.geometry.uuid, object.geometry)
            # self.materials.add(object.material.uuid, object.material)
        elif action == "remove":
            self.objects.remove(object.uuid)
            # self.geometries.remove(object.geometry.uuid)
            # self.materials.remove(object.material.uuid)


class Object3D:
    def __init__(self, uuid, type, geometry, material, **kwargs):
        self.uuid = uuid
        self.type = type
        self.geometry = geometry
        self.material = material
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_json(self):
        data = {
            "uuid": self.uuid,
            "type": self.type,
            "geometry": self.geometry.uuid,
            "material": self.material.uuid,
        }
        return data

    def __repr__(self):
        return f"<Object3D {self.uuid} {self.type}>"


class Material:
    def __init__(self, uuid, type, **kwargs):
        self.uuid = uuid
        self.type = type
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_json(self):
        data = {
            "uuid": self.uuid,
            "type": self.type,
        }
        return data

    def __repr__(self):
        return f"<Material {self.uuid} {self.type}>"


class Geometry:
    def __init__(self, uuid, type, **kwargs):
        self.uuid = uuid
        self.type = type
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_json(self):
        data = {
            "uuid": self.uuid,
            "type": self.type,
        }
        return data

    def __repr__(self):
        return f"<Geometry {self.uuid} {self.type}>"
