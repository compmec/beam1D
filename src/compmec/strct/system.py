from multiprocessing.sharedctypes import Value
import numpy as np
from compmec.strct.__classes__ import Structural1D
from compmec.strct.beam import Beam
from compmec.strct.solver import solve
    


class StructuralSystem(object):
    
    def __init__(self):
        self._elements = []
        self._BCs = []
        self._forces = []

    @property
    def elements(self):
        return self._elements

    @property
    def points(self):
        return self._points

    @elements.setter
    def elements(self, value):
        if not isinstance(value, (tuple, list)):
            raise TypeError("To set elements, it must be a list/tuple")
        for item in value:
            if not isinstance(item, Structural1D):
                raise TypeError("Every item inside elements must be a Structural1D instance")
        self._elements = value

    @points.setter
    def points(self, value):
        value = np.array(value)
        if value.ndim != 2:
            raise ValueError("Set points must have shape (npts, dim)")
        npts, dim = value.shape
        if not 0 < dim < 4:
            raise ValueError("The dimension must be 1, 2 or 3")
        self._points = value

    def add_element(self, value):
        if not isinstance(value, Structural1D):
            raise TypeError("To add an element, it must be a Structural 1D instance")
        self._elements.append(value)

    def add_BC(self, point:int, values:dict):
        if isinstance(point, int):
            npts = self.points.shape[0]
            if point >= npts:
                raise ValueError(f"You requested the point {point}, but there's only {npts} points")
        elif isinstance(point, tuple):
            point = self.point_at(point)
        else:
            raise TypeError("The point must be")
        if not isinstance(values):
            raise TypeError(f"The values must be a 'dict' not {type(values)}")
        for key, value in values.items():
            if not isinstance(key, str):
                raise TypeError(f"All the keys in the dictionary must be 'str' not {type(key)}")
            if key not in ("ux", "uy", "uz", "tx", "ty", "tz"):
                raise ValueError(f"The keys of dict must be 'ux', 'uy', 'uz', 'tx', 'ty' or 'tz', received {key}")
            value = float(value)

        for key, value in values.items():
            bcposition = self.__key2pos(key)
            self._BCs.append( (point, bcposition, value) )

    def __key2pos(self, key):
        dofs = self.__get_dofs()
        if key == "ux":
            return 0
        if key == "uy":
            if dofs == 1:
                raise ValueError("uy is not available when dofs = 1")
            return 1
        if key == "uz":
            if dofs == 2:
                raise ValueError("uz is not available when dofs = 1")
            return 2
        if key == "tx" or key == "ty" or key == "tz":
            if dofs != 6:
                raise ValueError("tx is not available when dofs != 6")
        if key == "tx":
            return 3
        if key == "ty":
            return 4
        if key == "tz":
            return 5
        raise ValueError(f"Got here, key is {key} and not found a position")

    def point_at(self, point:tuple, tolerance=1e-6):
        distsquare = np.array([(pi - point)**2 for pi in self.points])
        mindistsquare = np.min(distsquare)
        index = np.where(distsquare == mindistsquare)

    def run(self):
        K = self.__mount_K()
        F = self.__mount_F()
        U = self.__mount_U()
        U, F = solve()


    def __get_dofs(self):
        dim = self.points.shape[1]
        thereisbeam = False
        for element in self.elements:
            if isinstance(element, Beam):
                thereisbeam = True
                break
        if dim == 1:
            return 1
            # There's no beam, only truss and cable
        if dim == 2:
            if thereisbeam:
                return 3
            return 2
        if dim == 3:
            if thereisbeam:
                return 6
            return 3

    def __mount_U(self):
        npts = self.points.shape[0]
        dofs = self.__get_dofs()
        U = np.empty((npts, dofs), dtype="object")
        for i, j, v in self._BCs:
            U[i, j] = v
        return U

    def __mount_F(self):
        npts = self.points.shape[0]
        dofs = self.__get_dofs()
        F = np.zeros((npts, dofs))
        return F

    def __mount_K(self):
        npts = self.points.shape[0]
        dofs = self.__get_dofs()
        K = np.zeros((npts, dofs, npts, dofs))
        return K
