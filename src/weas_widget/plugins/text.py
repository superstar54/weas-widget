from ..base_class import WidgetWrapper


class Text(WidgetWrapper):

    catalog = "text"

    _attribute_map = {
        "settings": "text",
    }

    _extra_allowed_attrs = []

    def __init__(self, _widget):
        super().__init__(_widget)
