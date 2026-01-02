from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable, Iterable, List, Literal, Optional, Sequence

from .editing_tools import build_editing_tools
from .operations_tools import build_operations_tools
from .selection_tools import build_selection_tools
from .structure_tools import build_structure_tools
from .style_tools import build_style_tools
from .visualization_tools import build_visualization_tools


class WeasToolkit:
    """
    Build a collection of LangChain-compatible tools for driving a WeasWidget.

    Extensions
    ----------
    You can extend the toolkit in two ways:

    1) Register directly:

        toolkit = WeasToolkit(viewer)
        toolkit.register(my_tool)  # a langchain_core.tools.BaseTool
        toolkit.register(lambda viewer: [tool_a(viewer), tool_b(viewer)])  # factories are supported

    2) Plugin entry points:
       Provide entry points under the group ``weas_widget.tools``.
       Each entry point should resolve to either:
         - a tool instance, or
         - a callable returning a tool or list of tools (optionally accepting ``viewer``).
    """

    def __init__(
        self,
        viewer: Any,
        *,
        extra_tools: Optional[Sequence[Any]] = None,
        load_entry_points: bool = True,
        entry_point_group: str = "weas_widget.tools",
    ) -> None:
        self.viewer = viewer
        self._tools = self._build_tools()
        if load_entry_points:
            self._extend_from_entry_points(entry_point_group)
        if extra_tools:
            self.extend(extra_tools)

    @property
    def tools(self):
        return list(self._tools)

    def extend(
        self,
        tools: Sequence[Any],
        *,
        on_conflict: Literal["skip", "replace", "error"] = "skip",
    ) -> None:
        resolved = list(_resolve_tool_specs(tools, self.viewer))
        self._merge_tools(resolved, on_conflict=on_conflict, source="extra_tools")

    def register(
        self, tool: Any, *, on_conflict: Literal["skip", "replace", "error"] = "skip"
    ) -> None:
        self.extend([tool], on_conflict=on_conflict)

    def _merge_tools(
        self,
        new_tools: Sequence[Any],
        *,
        on_conflict: Literal["skip", "replace", "error"],
        source: str,
    ) -> None:
        existing = {getattr(t, "name", None): i for i, t in enumerate(self._tools)}
        for t in new_tools:
            name = getattr(t, "name", None)
            if not name:
                raise TypeError(f"Invalid tool from {source}: missing .name")
            if name in existing:
                if on_conflict == "skip":
                    warnings.warn(
                        f"Skipping duplicate tool name {name!r} from {source}.",
                        RuntimeWarning,
                    )
                    continue
                if on_conflict == "error":
                    raise ValueError(f"Duplicate tool name {name!r} from {source}.")
                self._tools[existing[name]] = t
            else:
                existing[name] = len(self._tools)
                self._tools.append(t)

    def _extend_from_entry_points(self, group: str) -> None:
        for spec in _load_tool_entry_points(group):
            try:
                resolved = list(_resolve_tool_specs([spec], self.viewer))
            except Exception as e:
                warnings.warn(
                    f"Failed to load tool plugin from entry point group {group!r}: {e}",
                    RuntimeWarning,
                )
                continue
            self._merge_tools(
                resolved, on_conflict="skip", source=f"entry_points:{group}"
            )

    def _build_tools(self):
        from weas_widget import WeasWidget

        viewer: WeasWidget = self.viewer
        return [
            *build_style_tools(viewer),
            *build_selection_tools(viewer),
            *build_structure_tools(viewer),
            *build_editing_tools(viewer),
            *build_operations_tools(viewer),
            *build_visualization_tools(viewer),
        ]


def create_weas_tools(
    viewer: Any,
    *,
    extra_tools: Optional[Sequence[Any]] = None,
    load_entry_points: bool = True,
    entry_point_group: str = "weas_widget.tools",
):
    """Helper returning `WeasToolkit(...).tools` with optional extensions."""
    return WeasToolkit(
        viewer,
        extra_tools=extra_tools,
        load_entry_points=load_entry_points,
        entry_point_group=entry_point_group,
    ).tools


def _is_tool_instance(obj: Any) -> bool:
    name = getattr(obj, "name", None)
    if not isinstance(name, str) or not name:
        return False
    return any(
        callable(getattr(obj, attr, None))
        for attr in ("invoke", "ainvoke", "run", "arun")
    )


def _call_maybe_with_viewer(factory: Callable[..., Any], viewer: Any) -> Any:
    try:
        return factory(viewer)
    except TypeError as e:
        try:
            sig = inspect.signature(factory)
        except Exception:
            raise
        if len(sig.parameters) == 0:
            return factory()
        raise e


def _flatten(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[Any] = []
        for x in value:
            out.extend(list(_flatten(x)))
        return out
    return [value]


def _resolve_tool_specs(specs: Sequence[Any], viewer: Any) -> Iterable[Any]:
    for spec in specs:
        for item in _flatten(spec):
            if item is None:
                continue
            if _is_tool_instance(item):
                yield item
                continue
            if callable(item):
                produced = _call_maybe_with_viewer(item, viewer)
                for t in _flatten(produced):
                    if not _is_tool_instance(t):
                        raise TypeError(f"Tool factory produced a non-tool: {t!r}")
                    yield t
                continue
            raise TypeError(f"Unsupported tool spec: {item!r}")


def _load_tool_entry_points(group: str) -> List[Any]:
    try:
        from importlib import metadata as importlib_metadata
    except Exception:
        import importlib_metadata

    eps = importlib_metadata.entry_points()
    if hasattr(eps, "select"):  # py>=3.10
        selected = list(eps.select(group=group))
    else:
        selected = list(eps.get(group, []))

    out: List[Any] = []
    for ep in selected:
        try:
            out.append(ep.load())
        except Exception as e:
            warnings.warn(
                f"Failed to import tool entry point {ep!r}: {e}", RuntimeWarning
            )
    return out
