import numpy as np
import foo as f


class Bar:
    def __init__(self, v):
        self.v = np.array(v)

    def isNull(self):
        return f.Foo(np.less(np.linalg.norm(self.v, axis=1), 1e-9))

