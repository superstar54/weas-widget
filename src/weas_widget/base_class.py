import difflib
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


class ChangeTrackingDict(dict):
    def __init__(self, *args, widget=None, key=None, parent=None, **kwargs):
        self._changed = False
        self._parent = parent
        self._widget = widget
        self._key = key
        super().__init__(*args, **kwargs)

        # Wrap any nested dictionaries
        for k, value in self.items():
            if isinstance(value, dict):
                nested_dict = ChangeTrackingDict(value, parent=self)
                super().__setitem__(k, nested_dict)
        self._mark_changed()

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, ChangeTrackingDict):
            value = ChangeTrackingDict(value, parent=self)
            value._parent = self
        super().__setitem__(key, value)
        self._mark_changed()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._mark_changed()

    def clear(self):
        super().clear()
        self._mark_changed()

    def pop(self, key, default=None):
        data = super().pop(key, default)
        self._mark_changed()
        return data

    def popitem(self):
        data = super().popitem()
        self._mark_changed()
        return data

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._mark_changed()

    def _mark_changed(self):
        """Set the changed flag to True, notify parent, and update widget if set."""
        self._changed = True
        if self._parent:
            self._parent._mark_changed()
        else:
            if not (self._widget and self._key):
                raise ValueError("Widget and key must be set to update the widget.")
            setattr(self._widget, self._key, self.as_dict())

    def has_changed(self):
        return self._changed

    def reset_changed_flag(self):
        """Reset the changed flag in this and all nested ChangeTrackingDicts."""
        self._changed = False
        for value in self.values():
            if isinstance(value, ChangeTrackingDict):
                value.reset_changed_flag()

    def as_dict(self):
        """Recursively convert ChangeTrackingDict to a standard dict."""
        result = {}
        for key, value in self.items():
            if isinstance(value, ChangeTrackingDict):
                result[key] = value.as_dict()
            else:
                result[key] = value
        return result
