class ParamTracker:
    def __init__(self, params):
        self.params = params
        self.accessed_keys = set()

    def __getitem__(self, key):
        self.accessed_keys.add(key)
        
        return self.params[key]["init"]

    def get_keys(self):
        return self.accessed_keys