import pytest


def test_agent_tools_structure_edits():
    pytest.importorskip("langchain_core")

    from weas_widget import WeasWidget
    from weas_widget.agent.tools import create_weas_tools

    viewer = WeasWidget()
    tools = {t.name: t for t in create_weas_tools(viewer)}

    tools["load_molecule"].invoke({"name": "H2O"})
    atoms = viewer.to_ase()
    assert atoms.get_chemical_formula() in {"H2O", "OH2"}
    assert len(atoms) == 3

    tools["select_atoms"].invoke({"indices": [0]})
    assert viewer.avr.selected_atoms_indices == [0]

    tools["replace_atoms"].invoke({"symbol": "F"})
    atoms = viewer.to_ase()
    assert atoms.get_chemical_symbols()[0] == "F"

    before = atoms.get_positions().copy()
    tools["translate"].invoke({"vector": [1.0, 0.0, 0.0]})
    after = viewer.to_ase().get_positions()
    assert (after[0] - before[0]).tolist() == pytest.approx([1.0, 0.0, 0.0])

    tools["add_atom"].invoke({"symbol": "He", "x": 0.0, "y": 0.0, "z": 0.0})
    assert len(viewer.to_ase()) == 4

    tools["select_atoms"].invoke({"indices": [3]})
    tools["delete_atoms"].invoke({})
    assert len(viewer.to_ase()) == 3


def test_agent_tool_schemas_are_openai_compatible():
    pytest.importorskip("langchain_core")

    from weas_widget import WeasWidget
    from weas_widget.agent.tools import create_weas_tools

    viewer = WeasWidget()
    for tool in create_weas_tools(viewer):
        schema = tool.get_input_schema().model_json_schema()
        props = schema.get("properties", {})
        for name in ("repeat", "vector", "axis"):
            if name in props and props[name].get("type") == "array":
                assert (
                    "items" in props[name]
                ), f"{tool.name}.{name} missing 'items' in JSON schema"


def test_agent_tools_style_and_surface_workflow():
    pytest.importorskip("langchain_core")

    from weas_widget import WeasWidget
    from weas_widget.agent.tools import create_weas_tools

    viewer = WeasWidget()
    tools = {t.name: t for t in create_weas_tools(viewer)}

    out = tools["list_style_options"].invoke({})
    assert "viewer" in out["summary"]
    assert "cell" in out["summary"]

    tools["set_style"].invoke({"key": "model_style", "value": "Stick"})
    assert int(viewer.avr.model_style) == 3

    tools["load_fcc_surface"].invoke(
        {"symbol": "Pt", "miller": [1, 1, 1], "size": [2, 2, 3], "vacuum": 8.0}
    )
    slab = viewer.to_ase()
    n_slab = len(slab)
    assert n_slab > 0

    tools["append_molecule"].invoke({"name": "H2O"})
    atoms = viewer.to_ase()
    assert len(atoms) == n_slab + 3
    sel = list(viewer.avr.selected_atoms_indices)
    assert sel == list(range(n_slab, n_slab + 3))

    tools["place_selected_on_top"].invoke({"clearance": 2.0})
    atoms2 = viewer.to_ase()
    pos = atoms2.get_positions()
    surf = [i for i in range(len(atoms2)) if i not in set(sel)]
    z_top = float(pos[surf, 2].max())
    z_min_sel = float(pos[sel, 2].min())
    assert z_min_sel == pytest.approx(z_top + 2.0, abs=1e-7)
