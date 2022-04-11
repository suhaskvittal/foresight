"""
    author: Suhas Vittal
    date:   4 April 2022
"""

from qiskit.transpiler.layout import Layout

class HashedLayout(Layout):
    def __init__(self, input_dict=None):
        super().__init__(input_dict=input_dict)

    def __eq__(self, other):
        vd1 = self.get_virtual_bits()
        vd2 = other.get_virtual_bits()

        for v in vd1:
            if v not in vd2 or vd1[v] != vd2[v]:
                return False
        return True

    def __hash__(self):
        return self[0].__hash__() + self[1].__hash__()    

    def from_layout(layout):
        return HashedLayout(input_dict=layout.get_virtual_bits())
