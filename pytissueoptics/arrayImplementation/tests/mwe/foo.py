import numpy as np
import bar as b

class Foo:
    def __init__(self, v):
        self.v = np.array(v)

    def all(self):
        return np.all(self.v)

    def circularImports(self):
        return b.Bar([0,0,0])