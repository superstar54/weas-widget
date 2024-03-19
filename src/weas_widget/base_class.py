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
