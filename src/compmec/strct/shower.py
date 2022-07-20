from matplotlib import tri
from matplotlib import pyplot as plt
from compmec.strct.section import Circle, HollowCircle, ThinCircle
import numpy as np
from compmec.strct.system import StaticSystem
from typing import List, Tuple, Iterable, Optional
from compmec.strct.__classes__ import Structural1D
from compmec.nurbs import SplineCurve
from compmec.strct.postrait import splinecurve_element, elementsolution

class AxonometricProjector(object):

    names = ["xy", "xz", "yz", "parallel xy", "parallel xz", "parallel yz",
             "trimetric", "dimetric", "isometric", "axonometric custom"]

    def __init__(self, name: str):
        if name == "xy" or name == "parallel xy":
            self.horizontal = (1, 0, 0)
            self.vertical = (0, 1, 0)
        elif name == "xz" or name == "parallel xz":
            self.horizontal = (-1, 0, 0)
            self.vertical = (0, 0, 1)
        elif name == "yz" or name == "parallel yz":
            self.horizontal = (0, 1, 0)
            self.vertical = (0, 0, 1)
        self.horizontal = np.array(self.horizontal)
        self.vertical = np.array(self.vertical)

    def __call__(self, point3D: Tuple[float, float, float]) -> Tuple[float, float, float]:
        point3D = np.array(point3D)
        if point3D.ndim != 1:
            raise ValueError("Point3D must be a 1D-array")
        if len(point3D) != 3:
            raise ValueError("Point3D must have lenght = 3")
        horizontal = self.horizontal
        vertical = self.vertical
        if np.abs(np.inner(horizontal, vertical)) > 0.01:  # cos 82 degrees
            raise ValueError("The horizontal vector and the vertical are not perpendicular")
        normal = np.cross(horizontal, vertical)
        point3D -= np.inner(point3D, normal) * normal
        x = np.inner(point3D, horizontal)
        y = np.inner(point3D, vertical)
        return x, y        

    
class PerspectiveProjector(object):

    names = ["military", "cabinet", "cavalier", 
             "one-point", "two-point", "three-point", "perspective custom"]

    def __init__(self, name: str):
        raise NotImplementedError("Needs Implementation: TO DO")

    def __call__(self, point3D: Tuple[float, float, float]) -> Tuple[float, float]:
        """
        Receives a 3D point and transform it to 2D point
        """
        raise NotImplementedError("TO DO")

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
            raise TypeError(f"The received projectionname is type {type(projectionname)}, not 'str'")
        if projectionname in AxonometricProjector.names:
            self.projector = AxonometricProjector(projectionname)
        elif projectionname in PerspectiveProjector.names:
            self.projector = AxonometricProjector(projectionname)
        else:
            raise ValueError(f"The received projectionname is unknown. Must be in {Projector.axonometricnames+Projector.perspectivenames}")
            
    def __call__(self, point3D: Tuple[float, float, float]) -> Tuple[float, float]:
        """
        Receives a 3D point and transform it to 2D point
        """
        return self.projector(point3D)


class Shower(object):
    def __init__(self):
        pass


class ShowerStaticSystem(Shower):

    def __init__(self, system: StaticSystem):
        if not isinstance(system, StaticSystem):
            raise TypeError(f"The given system is {type(system)}, not a StaticSystem")
        super().__init__()
        self.__system = system
    
    def getonesplinecurve(self, element: Structural1D, deformed: bool=False):
        ts = element.ts
        points = element.points
        if not deformed:
            return splinecurve_element(4, ts, points)
        solution = elementsolution(self.__system, element)
        return splinecurve_element(4, ts, points, solution)
        
    def getallsplinecurves(self, deformed: bool=False) -> List[SplineCurve]:
        """
        This function get all spline curves
        If there are nelem elements, then there's nlem curves as return
        """
        curves = []
        for element in self.__system._structure.elements:
            newcurve = self.getonesplinecurve(element, deformed)
            curves.append(newcurve)
        return curves

    def plot2D(self, projector: str = "xy", field: Optional[str] = None, deformed: Optional[bool]=False, axes=None):
        all3Dcurves = self.getallsplinecurves(deformed)
        if axes is None:
            axes = plt.gca()
        projector = Projector(projector)
        npts = 10
        tplot = np.linspace(0, 1, npts)
        for curve in all3Dcurves:
            all3Dpoints = curve(tplot)
            all2Dpoints = np.zeros((npts, 2))
            for j, point3D in enumerate(all3Dpoints):
                all2Dpoints[j] = projector(point3D)
            axes.plot(all2Dpoints[:, 0], all2Dpoints[:, 1], color="k", label="original")

    def plot3D(self, fieldname: Optional[str] = None, deformed: Optional[bool]=False, axes=None):
        all3Dcurves = self.getallsplinecurves(deformed)
        if axes is None:
            plt.figure()
            axes = plt.gca()
        npts = 10
        tplot = np.linspace(0, 1, npts)
        if fieldname is not None:
            cmap = plt.get_cmap("bwr")
        for curve in all3Dcurves:
            all3Dpoints = curve(tplot)
            if fieldname is None:
                axes.plot(all3Dpoints[:, 0], all3Dpoints[:, 1], all3Dpoints[:, 2], color="k")
            else:
                raise NotImplementedError("Field is not yet implemented")
                axes.scatter(p[:, 0], p[:, 1], p[:, 2], cmap=cmap, c=fieldvalues)
    

def plot_mesh(mesh):
    points = mesh.points
    cells = mesh.cells
    for cell in cells:
        if "triangle" != cell.type:
            continue
        connections = cell.data
    
    plt.figure()
    for connection in connections:
        ps = [points[c] for c in connection]
        ps.append(ps[0])
        x = [pi[0] for pi in ps]
        y = [pi[1] for pi in ps]
        plt.plot(x, y, color="k")

def function(p):
    x, y, z = p
    return np.sin(2*np.pi*x) + y**2
    return float(np.sum(np.array(p)**2))

def show_section(function, mesh, axes=None):
    
    # plt.figure()
    # plot_mesh(mesh)

    points = mesh.points
    points = np.array(points)
    cells = mesh.cells
    for cell in cells:
        if "triangle" != cell.type:
            continue
        connections = cell.data
    connections = np.array(connections, dtype="int16")
    x = points[:, 0]
    y = points[:, 1]
    v = [function(p) for p in points]
    triangulation = tri.Triangulation(x, y, connections)
    plt.tricontourf(triangulation, v)

    

def main():
    pass
    # circle = Circle(R = 1, nu = 0.2)
    # circle = HollowCircle(Ri=0.5, Re=1.0, nu=0.2)
    circle = ThinCircle(R=1, nu=0.2)
    mesh = circle.mesh()
    show_section(function, mesh)

    plt.show()


if __name__ == "__main__":
    main()
