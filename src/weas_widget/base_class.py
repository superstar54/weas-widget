import difflib


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


class Collection:
    def __init__(self, _widget):
        self._widget = _widget
        self._items = []
        self._key_map = {}

    def keys(self):
        return self._key_map.keys()

    def add(self, key, item):
        if key in self._key_map:
            # Replace the existing item at the same key
            index = self._key_map[key]
            self._items[index] = item
        else:
            # Add a new item and update the key map
            self._key_map[key] = len(self._items)
            self._items.append(item)

    def remove(self, key):
        # if key is an object with uuid
        if hasattr(key, "uuid"):
            key = key.uuid
        if key in self._key_map:
            index = self._key_map[key]
            # Remove the item and key map entry
            del self._items[index]
            del self._key_map[key]
            # Update indices in the key map
            for k, v in self._key_map.items():
                if v > index:
                    self._key_map[k] = v - 1
            self._widget.send_js_task({"name": "tjs.scene.remove", "args": [key]})

    def __getitem__(self, key):
        if isinstance(key, int):
            # Direct index access
            return self._items[key]
        elif isinstance(key, str):
            # Access by key
            index = self._key_map[key]
            return self._items[index]
        else:
            raise KeyError("Key must be an integer or string")

    def __delitem__(self, key):
        self.remove(key)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __repr__(self):
        return repr(self._items)

    def to_json(self):
        pass

    def from_json(self, json_data):
        pass
