from ..base_class import WidgetWrapper
import numpy as np


class LatticePlane(WidgetWrapper):

    catalog = "lattice_plane"

    _attribute_map = {
        "planes": "anyMesh",
    }

    _extra_allowed_attrs = ["settings"]

    def __init__(self, _widget):
        super().__init__(_widget)
        self.settings = {}

    @property
    def atoms(self):
        return self._widget.atoms

    @property
    def cell(self):
        return np.array(self.atoms["cell"]).reshape(3, 3)

    @property
    def cell_edges(self):
        """Edges of the cell"""
        edge_indices = [
            [0, 3],
            [0, 1],
            [4, 2],
            [4, 1],
            [3, 5],
            [2, 6],
            [7, 5],
            [7, 6],
            [0, 2],
            [3, 6],
            [1, 5],
            [4, 7],
        ]
        basis = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ]
        )
        positions = np.dot(basis, self.cell)
        edges = []
        for indices in edge_indices:
            edges.append(positions[indices])
        return edges

    @property
    def cell_volume(self):
        return np.dot(self.cell[0], np.cross(self.cell[1], self.cell[2]))

    @property
    def cell_reciprocal(self):
        from math import pi

        b1 = 2 * pi / self.cell_volume * np.cross(self.cell[1], self.cell[2])
        b2 = 2 * pi / self.cell_volume * np.cross(self.cell[2], self.cell[0])
        b3 = 2 * pi / self.cell_volume * np.cross(self.cell[0], self.cell[1])
        return np.array([b1, b2, b3])

    def add_plane_from_indices(self, name, indices, distance=1.0, **kwargs):
        """Add a plane setting from indices"""
        cell_reciprocal = self.cell_reciprocal
        normal = np.dot(indices, cell_reciprocal)
        normal = normal / np.linalg.norm(normal)
        point = distance * normal
        self.settings[name] = {
            "normal": normal,
            "point": point,
        }
        self.settings[name].update(kwargs)

    def add_plane_from_selected_atoms(self, name, **kwargs):
        """
        Add a plane setting from selected atoms
        """
        indices = self._widget.selectedAtomsIndices
        if len(indices) != 3:
            raise ValueError("Please select three atoms.")
        positions = np.array(self.atoms["positions"])
        center = np.mean(positions[indices], axis=0)
        normal = np.cross(
            positions[indices[1]] - positions[indices[0]],
            positions[indices[2]] - positions[indices[0]],
        )
        self.settings[name] = {
            "normal": normal,
            "point": center,
        }
        self.settings[name].update(kwargs)

    def draw(self, no=None):
        """Draw plane
        no: int
            spacegroup of structure, if None, no will be determined by
            get_spacegroup_number()
        include_center: bool
            include center of plane in the mesh
        """

        # TODO delete old plane
        if no is not None:
            self.no = no
        planes = self.build_plane(self.cell)
        for name, plane in planes.items():
            self.send_js_task(
                {
                    "name": "drawPlane",
                    "kwargs": {"plane": plane},
                }
            )

    def build_plane(self):
        """
        Build vertices, edges and faces of plane.
        """
        cellEdges = self.cell_edges
        planes = {}
        for name, data in self.settings.items():
            intersect_points = []
            # get intersection point
            for line in cellEdges:
                intersect_point = linePlaneIntersection(
                    line, data["normal"], data["point"]
                )
                if intersect_point is not None:
                    intersect_points.append(intersect_point)
                # get verts, edges, faces by Hull
            if len(intersect_points) < 3:
                continue
            vertices, edges, faces = faces_from_vertices(
                intersect_points, data["normal"], scale=data.get("scale", 1)
            )
            planes[name] = self.get_plane_data(vertices, edges, faces, data)
        self.planes = list(planes.values())

    def get_plane_data(self, vertices, edges, faces, plane):
        """
        build edge
        """
        if len(faces) > 0:
            plane.update(
                {
                    "vertices": vertices.reshape(-1).tolist(),
                    "edges": edges,
                    "faces": np.array(faces).reshape(-1).tolist(),
                    "edges_cylinder": {
                        "lengths": [],
                        "centers": [],
                        "normals": [],
                        "vertices": 6,
                        "color": (0.0, 0.0, 0.0, 1.0),
                        "width": plane.pop("width", 0.1),
                        "battr_inputs": {},
                    },
                }
            )
            for edge in edges:
                center = (vertices[edge[0]] + vertices[edge[1]]) / 2.0
                vec = vertices[edge[0]] - vertices[edge[1]]
                length = np.linalg.norm(vec)
                nvec = vec / length
                plane["edges_cylinder"]["lengths"].append(length)
                plane["edges_cylinder"]["centers"].append(center)
                plane["edges_cylinder"]["normals"].append(nvec)
        return plane


def faces_from_vertices(vertices, normal, scale=[1, 1, 1]):
    """
    get faces from vertices
    """
    # remove duplicative point
    vertices = np.unique(vertices, axis=0)
    n = len(vertices)
    if n < 3:
        return vertices, [], []
    center = np.mean(vertices, axis=0)
    v1 = vertices[0] - center
    angles = [[0, 0]]
    normal = normal / (np.linalg.norm(normal) + 1e-6)
    for i in range(1, n):
        v2 = vertices[i] - center
        x = np.cross(v1, v2)
        c = np.sign(np.dot(x, normal))
        angle = np.arctan2(c, np.dot(v1, v2))
        angles.append([i, angle])
    # scale
    vec = vertices - center
    # length = np.linalg.norm(vec, axis = 1)
    # nvec = vec/length[:, None]
    vertices = center + np.array([scale]) * vec
    # search convex polyhedra
    angles = sorted(angles, key=lambda x: x[1])
    faces = []
    # change faces to triangle
    for i in range(1, n - 1):
        faces.append([angles[0][0], angles[i][0], angles[i + 1][0]])
    # get edges
    edges = []
    for i in range(n - 1):
        edges.append([angles[i][0], angles[i + 1][0]])
    return vertices, edges, faces


def linePlaneIntersection(line, normal, point):
    """
    3D Line Segment and Plane Intersection
    - Point
    - Line contained in plane
    - No intersection
    """
    d = np.dot(point, normal)
    normalLine = line[0] - line[1]
    a = np.dot(normalLine, normal)
    # No intersection or Line contained in plane
    if np.isclose(a, 0):
        return None
    # in same side
    b = np.dot(line, normal) - d
    if b[0] * b[1] > 0:
        return None
    # Point
    v = point - line[0]
    d = np.dot(v, normal) / a
    point = np.round(line[0] + normalLine * d, 6)
    return point
