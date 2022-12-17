from typing import Iterable, Optional

import matplotlib as mpl
import numpy as np

from compmec.strct.__classes__ import Shower, System
from compmec.strct.geometry import Point2D, Point3D


class AxonometricProjector(object):

    names = [
        "xy",
        "xz",
        "yz",
        "parallel xy",
        "parallel xz",
        "parallel yz",
        "trimetric",
        "dimetric",
        "isometric",
        "axonometric custom",
    ]

    def __init__(self, name: str):
        if name == "xy" or name == "parallel xy":
            self._horizontal = Point3D([1, 0, 0])
            self._vertical = Point3D([0, 1, 0])
        elif name == "xz" or name == "parallel xz":
            self._horizontal = Point3D([-1, 0, 0])
            self._vertical = Point3D([0, 0, 1])
        elif name == "yz" or name == "parallel yz":
            self._horizontal = Point3D([0, 1, 0])
            self._vertical = Point3D([0, 0, 1])
        else:
            raise NotImplementedError

    def __call__(self, point: Point3D) -> Point2D:
        point3D = Point3D(point)
        x = np.inner(point3D, self._horizontal)
        y = np.inner(point3D, self._vertical)
        return Point2D([x, y])


class PerspectiveProjector(object):

    names = [
        "military",
        "cabinet",
        "cavalier",
        "one-point",
        "two-point",
        "three-point",
        "perspective custom",
    ]

    def __init__(self, name: str):
        raise NotImplementedError

    def __call__(self, point: Point3D) -> Point2D:
        """
        Receives a 3D point and transform it to 2D point
        """
        raise NotImplementedError


class Projector(object):
    def __docs__(self):
        """
        This class makes the projection. The options are:
        Axonometric:
            "xy" | "parallel xy"
            "xz" | "parallel xz"
            "yz" | "parallel yz"
            "trimetric"
            "dimetric"
            "isometric"
        Perspective:
            "military"
            "cabinet"
            "cavalier"
            "one-point"
            "two-point"
            "three-point"

        For more details
            https://en.wikipedia.org/wiki/3D_projection
        """

    def __init__(self, projectionname: str):
        if not isinstance(projectionname, str):
            error_msg = f"The projectionname must be 'str', not {type(projectionname)}"
            raise TypeError(error_msg)
        if projectionname in AxonometricProjector.names:
            self.projector = AxonometricProjector(projectionname)
        elif projectionname in PerspectiveProjector.names:
            self.projector = PerspectiveProjector(projectionname)
        else:
            projnames = AxonometricProjector.names + PerspectiveProjector.names
            error_msg = f"The received projectionname '{projectionname}' is unknown.\n"
            error_msg += f"Must be in {projnames}"
            raise ValueError(error_msg)

    def __call__(self, point: Point3D) -> Point2D:
        """
        Receives a 3D point and transform it to 2D point
        """
        return self.projector(point)


class ShowerStaticSystem(Shower):
    def __init__(self, system: System):
        if not isinstance(system, System):
            error_msg = f"The given system is {type(system)}, not a System"
            raise TypeError(error_msg)
        super().__init__()
        if system._solution is None:
            raise ValueError("You must run the simulation before put it in Shower")
        self.__system = system

    def getAll2DPoints(
        self, tplot: Iterable[float], deformed: Optional[bool], projector: Projector
    ):
        all2Dpoints = []
        npts = len(tplot)
        for element in self.__system._structure.elements:
            curve = element.field("d") if deformed else element.field("p")
            element3Dpoints = curve.evaluate(tplot)
            element2Dpoints = np.zeros((npts, 2))
            for j, point3D in enumerate(element3Dpoints):
                element2Dpoints[j] = projector(point3D)
            all2Dpoints.append(element2Dpoints)
        return all2Dpoints

    def plot2D_notfield(
        self, tplot: Iterable[float], projector: Projector, deformed: bool, axes
    ):
        all2Dpoints = self.getAll2DPoints(tplot, deformed, projector)
        for element2Dpoints in all2Dpoints:
            axes.plot(
                element2Dpoints[:, 0],
                element2Dpoints[:, 1],
                color="k",
                label="original",
            )

    def plot2D_withfield(
        self, tplot: Iterable[float], projector, deformed: bool, fieldname: str, axes
    ):
        fig = mpl.pyplot.gcf()
        all2Dpoints = self.getAll2DPoints(tplot, deformed, projector)
        allfieldvalues = []
        tmed = (tplot[1:] + tplot[:-1]) / 2
        for element in self.__system._structure.elements:
            fieldcurve = element.field(fieldname)
            fieldvalues = fieldcurve.evaluate(tmed)
            allfieldvalues.append(fieldvalues)
        cmap = mpl.pyplot.get_cmap("viridis")  # viridis, plasma, jet
        minfield = np.min(allfieldvalues)
        maxfield = np.max(allfieldvalues)
        invnormalize = (
            1 if abs(maxfield - minfield) < 1e-3 else 1 / (maxfield - minfield)
        )
        norm = mpl.colors.Normalize(vmin=minfield, vmax=maxfield)
        for points2D, fieldvalues in zip(all2Dpoints, allfieldvalues):
            colors_ts = invnormalize * (fieldvalues - minfield)  # Normalize
            for i, c in enumerate(colors_ts):
                axes.plot(
                    points2D[i : i + 2, 0],
                    points2D[i : i + 2, 1],
                    color=cmap(c),
                    linewidth=3,
                )
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes,
            orientation="vertical",
            label=fieldname,
        )

    def plot2D(
        self,
        projector: str = "xy",
        fieldname: Optional[str] = None,
        deformed: Optional[bool] = False,
        axes=None,
    ):
        if axes is None:
            axes = mpl.pyplot.gca()
        projector = Projector(projector)
        npts = 65
        tplot = np.linspace(0, 1, npts)
        if fieldname is None:
            self.plot2D_notfield(tplot, projector, deformed, axes)
        else:
            self.plot2D_withfield(tplot, projector, deformed, fieldname, axes)

    def plot3D(
        self,
        fieldname: Optional[str] = None,
        deformed: Optional[bool] = False,
        axes=None,
    ):
        if axes is None:
            mpl.pyplot.figure()
            axes = mpl.pyplot.gca()
        npts = 65
        tplot = np.linspace(0, 1, npts)
        if fieldname is not None:
            cmap = mpl.pyplot.get_cmap("bwr")
        for element in self.__system._structure.elements:
            curve3D = element.field("d") if deformed else element.field("p")
            points3D = curve3D.evaluate(tplot)
            if fieldname is None:
                axes.plot(points3D[:, 0], points3D[:, 1], points3D[:, 2], color="k")
            else:
                fieldcurve = element.field(fieldname)
                fieldvalues = fieldcurve(tplot)
                axes.scatter(
                    points3D[:, 0],
                    points3D[:, 1],
                    points3D[:, 2],
                    cmap=cmap,
                    c=fieldvalues,
                )
