import numpy as np
from typing import Iterable, Type, Union, Tuple
from compmec.strct.__classes__ import Structural1DInterface, ComputeFieldBeamInterface
from compmec.strct.fields import ComputeFieldBeam
from compmec.strct.solver import solve


class Geometry1D(object):
    def __init__(self):
        self._dim = None
        self._points = None

    @property
    def dim(self):
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
        value = np.array(value, dtype="float64")
        if value.ndim != 2:
            raise ValueError("Set points must have shape (npts, dim)")
        self._points = value
    
    def create_point(self, point: tuple) -> int:
        """
        Returns the index of the new created point.
        If the point exists, it returns ValueError
        """
        if not isinstance(point, (tuple, np.ndarray)):
            raise TypeError("Point must be tuple")
        index = self.find_point(point)
        if index is not None:
            raise ValueError(f"The received point already exists at index {index}")
        return self._create_point(point)

    def _create_point(self, point: tuple) -> int:
        if self._points is None:
            self.dim = 3
            self.points = np.zeros((1, 3))
            self.points[0, :len(point)] = point
            return 0
        npts, dim = self.points.shape
        newpoints = np.zeros((npts+1, dim))
        newpoints[:-1, :] = self.points[:, :]
        newpoints[-1, :len(point)] = point
        self._points = newpoints
        return npts

    def find_point(self, point: tuple, tolerance: float=1e-6) -> int:
        """
        Given a point like (0.1, 3.1), it returns the index of this point.
        If the point is too far (bigger than tolerance), it returns None
        """
        if not isinstance(tolerance, float):
            raise TypeError("Tolerance to find point must be a float")
        if not isinstance(point, (tuple, np.ndarray)):
            raise TypeError("Point must be a tuple")
        if self.dim is None:
            return None
        return self._find_point(point, tolerance)

    def _find_point(self, point: tuple, tolerance: float) -> int:
        """
        Internal unprotected function. See docs of the original function
        """
        n = len(point)
        distsquare = np.array([sum((pi[:n] - point)**2) for pi in self.points], dtype="float64")
        mindistsquare = np.min(distsquare)
        if np.all(mindistsquare > tolerance):
            return None
        index = np.where(distsquare == mindistsquare)
        if len(index) > 1:
            raise ValueError("There's more than 1 point at the same position")
        if len(index[0]) > 1:
            raise ValueError("Not expected get here.")
        return int(index[0])

    def index_point_at(self, point: tuple) -> int:
        """
        If the point doesn't exist, a new one is created
        """
        index = self.find_point(point)
        if index is None:
            return self.create_point(point)
        return index

    def index_exists(self, index: int) -> bool:
        return index < self.npts

class StaticLoad(object):
    def __init__(self):
        self._loads = []

    @property
    def loads(self):
        return self._loads

    def key2pos(self, key: str) -> int:
        if not isinstance(key, str):
            raise TypeError(f"Key must be an string. Received {type(key)}")
        if key not in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]:
            raise ValueError(f"Received key is invalid: {key}. Must be Fx, Fy, Fz, Mx, My, Mz")
        return self._key2pos(key)

    def _key2pos(self, key: str) -> int:
        if key[0] == "F":
            pos = 0
        elif key[0] == "M":
            pos = 3        
        if key[1] == "x":
            return pos
        if key[1] == "y":
            return pos+1
        if key[1] == "z":
            return pos+2

    def _pos2key(self, pos: int) -> str:
        if pos == 0:
            return "Fx"
        elif pos == 1:
            return "Fy"
        elif pos == 2:
            return "Fz"
        elif pos == 3:
            return "Mx"
        elif pos == 4:
            return "My"
        elif pos == 5:
            return "Mz"

    def add_load(self, index: int, values: dict) -> None:
        if not isinstance(index, int):
            raise TypeError(f"Index must be an integer, not {type(index)}")    
        if not isinstance(values, dict):
            raise TypeError(f"Values must be a dictionary, not {type(values)}")
        for key, item in values.items():
            if not isinstance(key, str):
                raise TypeError(f"Every key in dictionary must be a string, not {type(key)}")
            if not isinstance(item, (float, int)):
                raise TypeError(f"Every item in dictionary must be a float, not {type(item)}")
        return self._add_load_at_index(index, values)

    def add_dist_load(self, indexs: Iterable[int], values: dict, Ls: Iterable[float]):
        if not isinstance(indexs, (tuple, list)):
            raise TypeError("Indexs must be tuple/list")
        for index in indexs:
            try:
                int(index)
            except Exception as e:
                raise TypeError(f"Cannot convert index into int. Type = {type(index)}")
        if not isinstance(values, dict):
            raise TypeError("Values must be a dict")
        if not isinstance(Ls, (tuple, list)):
            raise TypeError("Ls must be a tuple/list")
        for Li in Ls:
            try:
                float(Li)
            except Exception as e:
                raise TypeError(f"Cannot convert index into float. Type = {type(Li)}")
        self._add_dist_load(indexs, values, Ls)

    def _add_dist_load(self, indexs: Iterable[int], values: dict, Ls: Iterable[float]):
        npts = len(indexs)
        concload = np.zeros(npts)
        for key, values in values.items():
            position = self.key2pos(key)
            concload[:] = 0
            for i, Li in enumerate(Ls):
                qa, qb = values[i], values[i+1]
                concload[i] += Li*(2*qa+qb)/6
                concload[i+1] += Li*(qa+2*qb)/6
            for i, index in enumerate(indexs):
                self._loads.append((index, position, concload[i]))
        
    def _add_load_at_index(self, index: int, values: dict):
        for key, value in values.items():
            position = self.key2pos(key)
            self._loads.append((index, position, value))


class StaticBoundaryCondition(object):
    def __init__(self):
        self._BCs = []

    @property
    def bcvals(self):
        return self._BCs

    def key2pos(self, key: str) -> int:
        if not isinstance(key, str):
            raise TypeError(f"Key must be an string. Received {type(key)}")
        if key not in ["ux", "uy", "uz", "tx", "ty", "tz"]:
            raise ValueError(f"Received key is invalid: {key}. Must be ux, uy, uz, tx, ty, tz")
        return self._key2pos(key)

    def _key2pos(self, key: str) -> int:
        if key[0] == "u":
            pos = 0
        elif key[0] == "t":
            pos = 3        
        if key[1] == "x":
            return pos
        if key[1] == "y":
            return pos+1
        if key[1] == "z":
            return pos+2

    def _pos2key(self, pos: int) -> str:
        if pos == 0:
            return "ux"
        elif pos == 1:
            return "uy"
        elif pos == 2:
            return "uz"
        elif pos == 3:
            return "tx"
        elif pos == 4:
            return "ty"
        elif pos == 5:
            return "tz"

    def add_BC(self, index: int, values: dict):
        if not isinstance(values, dict):
            raise TypeError("Values must be dict")
        return self._add_BC(index, values)        

    def _add_BC(self, index: int, values: dict):
        for key, value in values.items():
            bcpos = self.key2pos(key)
            self._BCs.append( (index, bcpos, value) )



class StaticStructure(object):
    def __init__(self):
        self._elements = []

    @property
    def elements(self):
        return self._elements

    def add_element(self, value: Structural1DInterface) -> None:
        if not isinstance(value, Structural1DInterface):
            raise TypeError("To add an element, it must be a Structural 1D instance")
        return self._add_element(value)

    def _add_element(self, value: Structural1DInterface) -> None:
        self._elements.append(value)

class StaticSystem():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(StaticSystem, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self._geometry = Geometry1D()
        self._structure = StaticStructure()
        self._loads = StaticLoad()
        self._boundarycondition = StaticBoundaryCondition()
        self._solution = None
    
    def add_element(self, element: Structural1DInterface):
        self._structure.add_element(element)
        self.__getpointsfrom(element)


    def add_load(self, point: tuple, loads: dict):
        """
        Add a load in a specific point.
        Example:
            point = (1.0, 3.5, -2.0)
            system.add_load(point, {"Fx": -30})
        The available loads are combinations of ("F", "M") and ("x", "y", "z", "n", "v", "w").
        Example:
            Fx: Force in x direction
            Mn: Momentum in normal direction
        """
        index = self._geometry.index_point_at(point)
        self._loads.add_load(index, loads)

    def add_dist_load(self, element: Structural1DInterface, interval: Iterable[float], values: dict):
        """
        Add a distribueted load in a interval.
        Example:
            beamAB = EulerBernoulli([A, B])
            interval = (0.2, 0.5, 0.7)
            loadsFy = (10, -30, 30) 
            system.add_dist_load(beamAB, interval, {"Fy": loadsFy})
        All the values inside interval must be in [0, 1]
        The available loads are the same as 'add_load' function:
            ("Fx", "Fy", "Fz", "Fn", "Fv", "Fw",
             "Mx", "My", "Mz", "Mn", "Mv", "Mw")
        The quantities of the interval must be tha same as each load.
        The loads are linear defined. That means:
            At (0.5) the value of "Fy" is -30
            At (0.6) the value of "Fy" is 0
            At (0.7) the value of "Fy" is 30
        """
        if not isinstance(element, Structural1DInterface):
            raise TypeError("Element must be Structural1DInterface")
        try:
            for t in interval:
                float(t)
        except Exception as e:
            raise TypeError("Interval must be a tuple of floats")
        
        interval, values = self.__compute_dist_points(element.ts, interval, values)
        points3D = np.array([element.path(t) for t in interval], dtype="float64")
        npts, dim = points3D.shape
        indexs = [self._geometry.index_point_at(point) for point in points3D]
        Ls = [np.linalg.norm(points3D[i+1]-points3D[i]) for i in range(npts-1)]
        self._loads.add_dist_load(indexs, values, Ls)

    def __compute_dist_points(self, ts: Iterable[float], interval: Iterable[float], values: dict):
        """
        There's an element with the ts values:
            ts = [0, 0.2, 0.4, 0.6, 0.8, 1]
        And then there are the inteval values like
            interval = [0.1, 0.5, 0.7]
        Then we want to make a new interval
            newinterval = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7]
        And we have to adapt also the new values, at forces.
        If we had
            values = {"Fx": [2, 10, 6]}
        Then we want
            newvalues = {"Fx": [2, 4, 8, 10, 8, 6]}
        """
        ts = np.array(ts)
        interval = list(interval)
        newinterval = interval.copy()
        mask = (ts - min(interval))*(max(interval) - ts) > 0
        newinterval.extend(ts[mask])
        newinterval.sort()
        npts = len(newinterval)
        for key, vals in values.items():
            newvals = np.zeros(npts)
            for i, t in enumerate(newinterval):
                if t in interval:
                    indvalue = interval.index(t)
                    newvals[i] = vals[indvalue]
                else:
                    indvalue = 0
                    while not (interval[indvalue] < t < interval[indvalue+1]):
                        indvalue += 1 
                    ta, tb = interval[indvalue], interval[indvalue+1]
                    qa, qb = vals[indvalue], vals[indvalue+1]
                    newvals[i] = (qa*(tb-t) + qb*(t-ta))/(tb-ta)
            values[key] = newvals
        return newinterval, values
        

    def add_BC(self, point: tuple, bcvals: dict):
        index = self._geometry.index_point_at(point)
        self._boundarycondition.add_BC(index, bcvals)
        
    @property
    def solution(self):
        if self._solution is None:
            raise ValueError("You must run the simulation before getting the solution values")
        return self._solution

    def __getpointsfrom(self, element: Structural1DInterface):
        for t in element.ts:
            p = element.path(t)
            self._geometry.index_point_at(p)

    def mount_U(self) -> np.ndarray:
        npts = self._geometry.npts
        U = np.empty((npts, 6), dtype="object")
        for index, position, displacement in self._boundarycondition.bcvals:
            U[index, position] = displacement
        return U

    def mount_F(self) -> np.ndarray:
        npts = self._geometry.npts
        F = np.zeros((npts, 6))
        for index, position, loads in self._loads.loads:
            F[index, position] += loads
        return F

    def mount_K(self) -> np.ndarray:
        npts = self._geometry.npts
        K = np.zeros((npts, 6, npts, 6))
        for element in self._structure.elements:
            Kloc = element.stiffness_matrix()
            inds = []
            for t in element.ts:
                searchpoint = element.path(t)
                newind = self._geometry.find_point(searchpoint)
                inds.append(newind)
            for i, indi in enumerate(inds):
                for j, indj in enumerate(inds):
                    K[indi, :, indj, :] += Kloc[i, :, j, :]
        return K

    def run(self):
        for element in self._structure.elements:
            self.__getpointsfrom(element)
        K = self.mount_K()
        F = self.mount_F()
        U = self.mount_U()
        U, F = solve(K, F, U)
        self._solution = U
        self.apply_on_elements()

    def apply_on_elements(self):
        for element in self._structure.elements:
            npts = len(element.ts)
            points = element.path(element.ts)
            indexs = np.zeros(npts, dtype="int32")
            for i, p in enumerate(points):
                indexs[i] = self._geometry.find_point(p)
            Uelem = np.zeros((npts, 6))
            for i, j in enumerate(indexs):
                Uelem[i, :] = self._solution[j, :]
            field = ComputeFieldBeam(element, Uelem)
            element.field = field


    
    
