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
