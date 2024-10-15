import difflib
from copy import deepcopy
from typing import Any, Dict, Optional
from functools import wraps


class WidgetWrapper:
    """A base class for widget-related functionalities.

    This class provides dynamic property creation, attribute suggestions, and custom task application.
    """

    catalog = "base"
    _attribute_map = {}
    _extra_allowed_attrs = []

    def __init__(self, _widget):
        object.__setattr__(self, "_widget", _widget)
        for attr, widget_attr in self._attribute_map.items():
            self._create_property(attr, widget_attr)
        self._widget.observe(self._apply_task, "python_task")

    def _apply_task(self, change):
        if not change["new"] or change["new"]["catalog"] != self.catalog:
            return
        action = change["new"]["action"]
        setattr(self, action, change["new"]["data"])

    def _create_property(self, attribute_name, widget_attribute):
        def getter(self):
            return getattr(self._widget, widget_attribute)

        def setter(self, value):
            setattr(self._widget, widget_attribute, value)

        setattr(self.__class__, attribute_name, property(getter, setter))

    @property
    def all_attributes(self):
        return (
            list(self._attribute_map.keys())
            + ["_widget", "_attributes"]
            + self._extra_allowed_attrs
        )

    def _suggest_attribute(self, name):

        suggestions = difflib.get_close_matches(name, self.all_attributes)
        if suggestions:
            return f" Did you mean: {', '.join(suggestions)}?"
        else:
            return f" Available attributes are: {', '.join(self.all_attributes)}."

    def __setattr__(self, name, value):
        if name in self.all_attributes:
            object.__setattr__(self, name, value)
        else:
            suggestion = self._suggest_attribute(name)
            raise AttributeError(
                f"'{name}' is not a valid attribute of '{self.__class__.__name__}'.{suggestion}"
            )


def with_on_change(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)  # Call the original method
        self._on_change()  # Trigger change tracking
        return result

    return wrapper


class DictWrapper:
    def __init__(self, widget: Any, key: str):
        self._widget = widget
        self._key = key

    @property
    def data(self) -> Dict:
        return getattr(self._widget, self._key)

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    @with_on_change
    def __setitem__(self, key: Any, value: Any):
        if self.data.get(key) != value:
            self.data[key] = value

    @with_on_change
    def __delitem__(self, key: Any):
        if key in self.data:
            del self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __repr__(self) -> str:
        return repr(self.data)

    def __contains__(self, key: Any) -> bool:
        return key in self.data

    def get(self, key: Any, default: Optional[Any] = None) -> Any:
        return self.data.get(key, default)

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    @with_on_change
    def update(self, other: Dict):
        if other:
            self.data.update(other)

    @with_on_change
    def pop(self, key: Any, default: Optional[Any] = None) -> Any:
        return self.data.pop(key, default)

    @with_on_change
    def popitem(self) -> tuple:
        return self.data.popitem()

    @with_on_change
    def clear(self):
        if self.data:
            self.data.clear()

    def _on_change(self, value=None):
        print("on_change")
        if value is not None:
            setattr(self._widget, self._key, value)
        else:
            setattr(self._widget, self._key, deepcopy(self.data))
