from dataclasses import dataclass

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    # Alias para width
    @property
    def w(self):
        return self.width

    @w.setter
    def w(self, value):
        self.width = value

    # Alias para height
    @property
    def h(self):
        return self.height

    @h.setter
    def h(self, value):
        self.height = value