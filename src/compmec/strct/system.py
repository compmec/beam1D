from typing import Iterable, Tuple, Type, Union

import numpy as np

from compmec.strct.__classes__ import Element1D, Point
from compmec.strct.fields import ComputeFieldBeam
from compmec.strct.geometry import Geometry1D
from compmec.strct.solver import solve


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
            raise ValueError(
                f"Received key is invalid: {key}. Must be Fx, Fy, Fz, Mx, My, Mz"
            )
        return self._key2pos(key)

    def _key2pos(self, key: str) -> int:
        if key[0] == "F":
            pos = 0
        elif key[0] == "M":
            pos = 3
        if key[1] == "x":
            return pos
        if key[1] == "y":
            return pos + 1
        if key[1] == "z":
            return pos + 2

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
                raise TypeError(
                    f"Every key in dictionary must be a string, not {type(key)}"
                )
            if not isinstance(item, (float, int)):
                raise TypeError(
                    f"Every item in dictionary must be a float, not {type(item)}"
                )
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
                qa, qb = values[i], values[i + 1]
                concload[i] += Li * (2 * qa + qb) / 6
                concload[i + 1] += Li * (qa + 2 * qb) / 6
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
            raise ValueError(
                f"Received key is invalid: {key}. Must be ux, uy, uz, tx, ty, tz"
            )
        return self._key2pos(key)

    def _key2pos(self, key: str) -> int:
        if key[0] == "u":
            pos = 0
        elif key[0] == "t":
            pos = 3
        if key[1] == "x":
            return pos
        if key[1] == "y":
            return pos + 1
        if key[1] == "z":
            return pos + 2

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
            self._BCs.append((index, bcpos, value))


class StaticStructure(object):
    def __init__(self):
        self._elements = []

    @property
    def elements(self):
        return self._elements

    def add_element(self, value: Element1D) -> None:
        if not isinstance(value, Element1D):
            raise TypeError("To add an element, it must be a Structural 1D instance")
        return self._add_element(value)

    def _add_element(self, value: Element1D) -> None:
        self._elements.append(value)


class StaticSystem:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(StaticSystem, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self._geometry = Geometry1D()
        self._structure = StaticStructure()
        self._loads = StaticLoad()
        self._boundarycondition = StaticBoundaryCondition()
        self._solution = None

    def add_element(self, element: Element1D):
        self._structure.add_element(element)

    def add_load(self, point: Point, loads: dict):
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
        index = self._geometry.find_point(point)
        if index is None:
            index = self._geometry.create_point(point)
        self._loads.add_load(index, loads)

    def add_dist_load(
        self, element: Element1D, interval: Iterable[float], values: dict
    ):
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
        if not isinstance(element, Element1D):
            raise TypeError("Element must be Element1D")
        try:
            for t in interval:
                float(t)
        except Exception as e:
            raise TypeError("Interval must be a tuple of floats")

        interval, values = self.__compute_dist_points(element.ts, interval, values)
        points3D = np.array([element.path(t) for t in interval], dtype="float64")
        npts, dim = points3D.shape
        indexs = []
        for point in points3D:
            index = self._geometry.find_point(point)
            if index is None:
                index = self._geometry.create_point(point)
            indexs.append(index)
        Ls = [np.linalg.norm(points3D[i + 1] - points3D[i]) for i in range(npts - 1)]
        self._loads.add_dist_load(indexs, values, Ls)

    def __compute_dist_points(
        self, ts: Iterable[float], interval: Iterable[float], values: dict
    ):
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
        mask = (ts - min(interval)) * (max(interval) - ts) > 0
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
                    while not (interval[indvalue] < t < interval[indvalue + 1]):
                        indvalue += 1
                    ta, tb = interval[indvalue], interval[indvalue + 1]
                    qa, qb = vals[indvalue], vals[indvalue + 1]
                    newvals[i] = (qa * (tb - t) + qb * (t - ta)) / (tb - ta)
            values[key] = newvals
        return newinterval, values

    def add_BC(self, point: Point, bcvals: dict):
        index = self._geometry.find_point(point)
        if index is None:
            index = self._geometry.create_point(point)
        self._boundarycondition.add_BC(index, bcvals)

    @property
    def solution(self):
        if self._solution is None:
            raise ValueError(
                "You must run the simulation before getting the solution values"
            )
        return self._solution

    def __getpointsfrom(self, element: Element1D):
        for t in element.ts:
            p = element.path(t)
            index = self._geometry.find_point(p)
            if index is None:
                self._geometry.create_point(p)

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
            element.set_field(field)
