
class Units():
    def __init__(self, **kwargs):
        for key, value in units.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
        self.__dict__.update(kwargs)