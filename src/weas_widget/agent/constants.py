from typing import Dict, Tuple

MODEL_STYLE_MAP: Dict[str, int] = {
    "Ball": 0,
    "Ball + Stick": 1,
    "Polyhedra": 2,
    "Stick": 3,
    "Line": 4,
}

COLOR_TYPES: Tuple[str, ...] = ("JMOL", "VESTA", "CPK")
COLOR_BYS: Tuple[str, ...] = ("Element", "Index", "Random", "Uniform")
RADIUS_TYPES: Tuple[str, ...] = ("Covalent", "VDW")
MATERIAL_TYPES: Tuple[str, ...] = ("Standard", "Phong", "Basic")
ATOM_LABEL_TYPES: Tuple[str, ...] = ("None", "Symbol", "Index")
