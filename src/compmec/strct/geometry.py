from typing import Tuple, Union

import numpy as np

from compmec.strct.__classes__ import Point


class PointBase(Point):
    @staticmethod
    def validation_creation(point: tuple[float]):
        if not isinstance(point, (list, tuple, np.ndarray)):
            raise TypeError("Point3D must be created from list/tuple/numpy array")
        for i, pi in enumerate(point):
            if not isinstance(pi, (int, float)):
                error_msg = f"To search the point, every coordinate must be a float.\n"
                error_msg += f"    type(point[{i}]) = {type(pi)}"
                raise TypeError(error_msg)

    def __add__(self, other: tuple[float]):
        return self.__class__([pi + qi for pi, qi in zip(self, other)])

    def __sub__(self, other: tuple[float]):
        return self.__class__([pi - qi for pi, qi in zip(self, other)])

    def __radd__(self, other: tuple[float]):
        return self.__class__([pi + qi for pi, qi in zip(self, other)])

    def __rsub__(self, other: tuple[float]):
        return self.__class__([qi - pi for pi, qi in zip(self, other)])

    def __eq__(self, other: tuple[float]):
        if isinstance(other, self.__class__):
            pass
        elif not isinstance(other, (tuple, list, np.ndarray)):
            return False
        else:
            try:
                other = self.__class__(other)
            except Exception as e:
                return False
        for pi, qi in zip(self, other):
            if pi != qi:
                return False
        return True

    def __ne__(self, other: tuple[float]):
        return not self.__eq__(other)


class Point2D(PointBase, Tuple):
    @staticmethod
    def validation_creation(point: tuple[float]):
        PointBase.validation_creation(point)
        if len(point) != 2:
            error_msg = "Point2D must be created with two float values. len(point) = {len(point)}"
            raise ValueError(error_msg)

    def __new__(cls, point: tuple[float]):
        if isinstance(point, Point2D):
            return point
        Point2D.validation_creation(point)
        return super(Point2D, cls).__new__(cls, tuple(point))


class Point3D(PointBase, Tuple):

    all_indexed_instances = []

    @staticmethod
    def validation_creation(point: tuple[float]):
        PointBase.validation_creation(point)
        if len(point) != 3:
            error_msg = "Point3D must be created with three float values. len(point) = {len(point)}"
            raise ValueError(error_msg)

    def __new__(cls, point: tuple[float]):
        if isinstance(point, Point3D):
            return point
        Point3D.validation_creation(point)
        self = super(Point3D, cls).__new__(cls, tuple(point))
        self._femid = None
        return self

    def _find_index_within_tolerance(self, tolerance: float):
        indexs = []
        for i, point in enumerate(Point3D.all_indexed_instances):
            distance = sum([(pi - qi) ** 2 for pi, qi in zip(self, point)])
            if distance < tolerance:
                indexs.append(i)
        return indexs

    @property
    def femid(self) -> int:
        if self._femid is not None:
            return self._femid
        indexs = self._find_index_within_tolerance(1e-6)
        if len(indexs) == 0:
            return None
        if len(indexs) > 1:
            raise ValueError("To get point, must have less than 2 points")
        self._femid = indexs[0]
        return self._femid

    def get_index(self):
        if self.femid is None:
            self.new_index()
        return self.femid

    def new_index(self):
        Point3D.all_indexed_instances.append(self)
        self._femid = len(Point3D.all_indexed_instances) - 1


class Geometry1D(object):
    def __init__(self):
        self._global_indexs = []

    @property
    def points(self):
        instances = Point3D.all_indexed_instances
        all_local_points = [instances[index] for index in self._global_indexs]
        return np.array(all_local_points)

    @property
    def npts(self):
        return len(self._global_indexs)

    def __contains__(self, point: Point3D):
        if self.find_point(point) is None:
            return False
        return True

    def find_point(
        self, point: Tuple[float], tolerance: float = 1e-6
    ) -> Union[int, None]:
        """
        Given a point like (0.1, 3.1, 5), it returns the index of this point.
        If the point is too far (bigger than tolerance), it returns None
        """
        if not isinstance(tolerance, (int, float)):
            raise TypeError("Tolerance to find point must be a float")
        if tolerance <= 0:
            raise ValueError("Tolerance must be positive!")
        if self.npts == 0:
            return None
        point = Point3D(point)
        return self._find_point(point, tolerance)

    def _find_point(self, point: Point3D, tolerance: float) -> Union[int, None]:
        """
        Internal unprotected function. See docs of the original function
        """
        distances = [sum((pi - point) ** 2) for pi in self.points]
        distsquare = np.array(distances, dtype="float64")
        mindistsquare = np.min(distsquare)
        if np.all(mindistsquare > tolerance**2):
            return None
        local_indexs = np.where(distsquare == mindistsquare)[0]
        if len(local_indexs) > 1:
            raise ValueError("There's more than 1 point at the same position")
        return int(local_indexs[0])

    def add_point(self, point: Tuple[float]) -> int:
        """
        Creates a new point, and add it into geometry.
        Returns the index of the new created point.
        If the point exists, it gives ValueError
        """
        point = Point3D(point)
        return self._add_point(point)

    def _add_point(self, point: Point3D) -> int:
        global_index = point.get_index()
        if global_index in self._global_indexs:
            local_index = self._global_indexs.index(global_index)
        else:
            self._global_indexs.append(global_index)
            local_index = len(self._global_indexs) - 1
        return local_index
