class Registry:
    def __init__(self, name):
        self._name = name
        self._dict = {}

    def __contains__(self, key):
        return key in self._dict

    def __getitem__(self, key):
        if key not in self._dict:
            raise KeyError(f"{key} is not registered in {self._name}")
        return self._dict[key]

    def register(self, key=None):
        if callable(key):
            name = key.__name__
            if name in self._dict:
                raise KeyError(f"{name} already registered in {self._name}")
            self._dict[name] = key
            return key

        def decorator(fn):
            name = key or fn.__name__
            if name in self._dict:
                raise KeyError(f"{name} already registered in {self._name}")
            self._dict[name] = fn
            return fn

        return decorator

    def get(self, key, default=None):
        return self._dict.get(key, default)

    def list_keys(self):
        return list(self._dict.keys())

    def __repr__(self):
        return f"{self._name}Registry({list(self._dict.keys())})"
