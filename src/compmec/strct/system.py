import numpy as np
from compmec.strct.__classes__ import Structural1D
from compmec.strct.solver import solve


class Geometry1D(object):

    def __init__(self):
        self._dim = None
        self._points = None

    @property
    def dim(self):
        if self._dim is None:
            raise ValueError("Dimension was not set")
        return self._dim

    @property
    def points(self):
        if self._points is None:
            raise ValueError("Points were requested, but None was found")
        return self._points

    @property
    def npts(self):
        return self.points.shape[0]

    @dim.setter
    def dim(self, value: int):
        if not isinstance(value, int):
            raise TypeError("Receveid dimension is not a integer: %s" % type(value))
        if not 0 < value < 4:
            raise ValueError("The dimension must be 1, 2 or 3")
        self._dim = value

    @points.setter
    def points(self, value: np.ndarray):
        if self._points is not None:
            raise ValueError("There are points. Cannot subcribe. Use add_point(point) instead")
        value = np.array(value)
        if value.ndim != 2:
            raise ValueError("Set points must have shape (npts, dim)")
        if self.dim != value.shape[1]:
            raise ValueError(f"Set structural dimension is {self.dim}. Received points has dimension %d. Must be equal" %(self.dim, value.shape[1]))
        self._points = value
    
    def create_point(self, point: tuple) -> int:
        """
        Returns the index of the new created point.
        If the point exists, it returns ValueError
        """
        if not isinstance(point, tuple):
            raise TypeError("Point must be tuple")
        index = self.find_point_at(point)
        if index is not None:
            raise ValueError(f"The received point already exists at index {index}")
        return self._create_point(point)

    def _create_point(self, point: tuple) -> int:
        npts, dim = self.points.shape
        newpoints = np.zeros((npts+1, dim))
        newpoints[:-1,:] = self.points[:, :]
        newpoints[-1, :] = point
        self.points = newpoints
        return npts

    def find_point_at(self, point:tuple, tolerance: float=1e-6) -> int:
        """
        Given a point like (0.1, 3.1), it returns the index of this point.
        If the point is too far (bigger than tolerance), it returns None
        """
        if not isinstance(tolerance, float):
            raise TypeError("Tolerance to find point must be a float")
        if not isinstance(point, tuple | list):
            raise TypeError("Point must be a tuple or a list of 3 elements")
        if len(point) != self.dim:
            raise ValueError(f"Point has len {len(point)}, but the dimension is {self.dim}")
        return self._find_point_at(point, tolerance)

    def _find_point_at(self, point:tuple, tolerance: float) -> int:
        """
        Internal unprotected function. See docs of the original function
        """
        distsquare = np.array([(pi - point)**2 for pi in self.points])
        mindistsquare = np.min(distsquare)
        if np.all(mindistsquare > tolerance):
            return None
        index = np.where(distsquare == mindistsquare)
        return index

    def index_point_at(self, point: int | tuple) -> int:
        if isinstance(point, int):
            return point
        index = self.find_point_at(point)
        if index is None:
            return self.create_point(point)

    def index_exists(self, index: int) -> bool:
        return index < self.npts

class StaticForce(object):
    def __init__(self, dim: int):
        self._charges = []

    def key2pos(self, key: str) -> int:
        if not isinstance(key, str):
            raise TypeError(f"Key must be an string. Received {type(key)}")
        if key not in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]:
            raise ValueError(f"Received key is invalid: {key}. Must be Fx, Fy, Fz, Mx, My, Mz")
        return self._key2pos(key)

    def _key2pos(self, key: str) -> int:
        if key == "F":
            pos = 0
        elif key == "M":
            pos = 3        
        if key[1] == "x":
            return pos
        if key[2] == "y":
            return pos+1
        if key[3] == "z":
            return pos+2

    def add_charge(self, point: int | tuple, values: dict) -> None:
        if not isinstance(values, dict):
            raise TypeError("Values must be a dictionary")
        if isinstance(point, int):
            if not Geometry1D.index_exists(point):
                raise ValueError(f"Index {point} of points doesn't exist!")
            return self._add_charge_at_index(point, values)
        elif isinstance(point, tuple):
            if isinstance(point[0], Structural1D):
                if len(point) != 2:
                    raise ValueError("When Structural1D, must pass 2 arguments")
                if not isinstance(point[1], float):
                    raise TypeError("The second argument must be a float")
                if position < 0 or 1 < position:
                    raise ValueError("The position value must be in [0, 1]")
                element = point[0]
                position = point[1]
                point = element.path(position)
            index = Geometry1D.index_point_at(point)
            return self._add_charge_at_index(index, values)
        else:
            raise TypeError("Point must be int or tuple")
        

    def _add_charge_at_index(self, index: int, values: dict):
        for key, value in values:
            position = self.key2pos(key)
            self._charges.append((index, position, value))

    def __mount_F(self, dofs: list) -> np.ndarray:
        F = np.zeros((Geometry1D.npts, len(dofs)))
        for i, j, c in self._charges:
            F[i, j] += c
        return F


class StaticBoundaryCondition(object):
    def __init__(self, dim: int):
        self._BCs = []

    def key2pos(self, key: str) -> int:
        if not isinstance(key, str):
            raise TypeError(f"Key must be an string. Received {type(key)}")
        if key not in ["ux", "uy", "uz", "tx", "ty", "tz"]:
            raise ValueError(f"Received key is invalid: {key}. Must be ux, uy, uz, tx, ty, tz")
        return self._key2pos(key)

    def _key2pos(self, key: str) -> int:
        if key == "u":
            pos = 0
        elif key == "t":
            pos = 3        
        if key[1] == "x":
            return pos
        if key[2] == "y":
            return pos+1
        if key[3] == "z":
            return pos+2

    def add_BC(self, point: int | tuple, values: dict):
        index = Geometry1D.index_point_at(point)
        if not isinstance(values, dict):
            raise TypeError("Values must be dict")
        return self._add_BC(self, index, values)        

    def _add_BC(self, index: int, values: dict):
        for key, value in values.items():
            bcpos = self.key2pos(key)
            self._BCs.append( (index, bcpos, value) )

    def __mount_U(self, dofs: list) -> np.ndarray:
        U = np.empty((Geometry1D.npts, len(dofs)), dtype="object")
        for i, j, u in self._BCs:
            U[i, j] = u
        return U

class StaticStructure(object):
    
    def __init__(self):
        self._elements = []

    @property
    def elements(self):
        return self._elements

    @elements.setter
    def elements(self, value: list | tuple):
        if len(self._elements):
            raise ValueError("There are elements. Cannot subscribe. Use add_element(element) instead")
        if not isinstance(value, (tuple, list)):
            raise TypeError("To set elements, it must be a list/tuple")
        for item in value:
            if not isinstance(item, Structural1D):
                raise TypeError("Every item inside elements must be a Structural1D instance")
        self._elements = value

    def add_element(self, value: Structural1D) -> None:
        if not isinstance(value, Structural1D):
            raise TypeError("To add an element, it must be a Structural 1D instance")
        return self._add_element(value)

    def _add_element(self, value: Structural1D) -> None:
        self._elements.append(value)

    def __mount_K(self, dofs: list) -> np.ndarray:
        npts = Geometry1D.npts
        K = np.zeros((npts, len(dofs), npts, len(dofs)))
        for element in self._elements:
            Kloc = element.stiffness_matrix()
            ind0 = Geometry1D.find_point_at(element.path(0))
            ind1 = Geometry1D.find_point_at(element.path(1))
            K[ind0, ind0] += Kloc[0, 0]
            K[ind0, ind1] += Kloc[0, 1]
            K[ind1, ind0] += Kloc[1, 0]
            K[ind1, ind1] += Kloc[1, 1]
        return K

class StaticSystem(Geometry1D, StaticForce, StaticBoundaryCondition, StaticStructure):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(StaticSystem, cls).__new__(cls)
        return cls.instance

    def __init__(self, dim: int):
        self.dim = dim

    def run(self):
        K = self.__mount_K()
        F = self.__mount_F()
        U = self.__mount_U()
        U, F = solve(K, F, U)
        self._solution = U

    
