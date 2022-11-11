from compmec.strct.section import Circle, HollowCircle, ThinCircle
import numpy as np
from compmec.strct.system import StaticSystem
from typing import List, Tuple, Iterable, Optional
from compmec.strct.__classes__ import Structural1DInterface
import matplotlib as mpl

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
        self.horizontal = np.array(self.horizontal, dtype="float64")
        self.vertical = np.array(self.vertical, dtype="float64")

    def __call__(self, point3D: Tuple[float, float, float]) -> Tuple[float, float, float]:
        point3D = np.array(point3D, dtype="float64")
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
    
    def getonesplinecurve(self, element: Structural1DInterface, deformed: bool=False):
        if not deformed:
            return element.path
        raise NotImplementedError("Deformed: Should be implemented")
    
    def getAll2DPoints(self, tplot: Iterable[float], deformed: Optional[bool], projector: Projector):
        all2Dpoints = []
        npts = len(tplot)
        for element in self.__system._structure.elements:
            element3Dpoints = element.evaluate(tplot, deformed)
            element2Dpoints = np.zeros((npts, 2))
            for j, point3D in enumerate(element3Dpoints):
                element2Dpoints[j] = projector(point3D)
            all2Dpoints.append(element2Dpoints)
        return all2Dpoints
    
    def plot2D_notfield(self, tplot: Iterable[float], projector: Projector, deformed: bool, axes):
        all2Dpoints = self.getAll2DPoints(tplot, deformed, projector)
        for element2Dpoints in all2Dpoints:
            axes.plot(element2Dpoints[:, 0], element2Dpoints[:, 1], color="k", label="original")
            
    def plot2D_withfield(self, tplot: Iterable[float], projector, deformed: bool, fieldname: str, axes):
        fig = mpl.pyplot.gcf()
        all2Dpoints = self.getAll2DPoints(tplot, deformed, projector)
        allfieldvalues = []
        tmed = (tplot[1:]+tplot[:-1])/2
        minfield = 1e+9
        maxfield = -1e+9
        for element in self.__system._structure.elements:
            fieldcurve = element.field(fieldname)
            fieldvalues = fieldcurve(tmed).reshape(-1)
            if np.min(fieldvalues) < minfield:
                minfield = np.min(fieldvalues)
            if np.max(fieldvalues) > maxfield:
                maxfield = np.max(fieldvalues)
            allfieldvalues.append(fieldvalues)
        if minfield >= 0:
            minfield = 0
            cmap = mpl.pyplot.get_cmap("viridis")  # viridis, plasma, jet
        else:
            cmap = mpl.pyplot.get_cmap("coolwarm")  # bwr, coolwarm
            maxfield = max(abs(minfield), maxfield)
            minfield = -maxfield
        if minfield == maxfield:
            if minfield == 0:
                maxfield = 1
            else:
                maxfield = abs(maxfield)
                minfield = -maxfield
        norm = mpl.colors.Normalize(vmin=minfield, vmax=maxfield)
        for points2D, fieldvalues in zip(all2Dpoints, allfieldvalues):
            colors_ts = (fieldvalues-minfield)/(maxfield-minfield)  # Normalize
            for i, c in enumerate(colors_ts):
                axes.plot(points2D[i:i+2, 0], points2D[i:i+2, 1], color=cmap(c), linewidth=3)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='vertical', label=fieldname)

    def plot2D(self, projector: str = "xy", fieldname: Optional[str] = None, deformed: Optional[bool]=False, axes=None):
        if axes is None:
            axes = mpl.pyplot.gca()
        projector = Projector(projector)
        npts = 65
        tplot = np.linspace(0, 1, npts)
        if fieldname is None:
            self.plot2D_notfield(tplot, projector, deformed, axes)
        else:
            self.plot2D_withfield(tplot, projector, deformed, fieldname, axes)
            
        

    
                

    def plot3D(self, fieldname: Optional[str] = None, deformed: Optional[bool]=False, axes=None):
        if axes is None:
            mpl.pyplot.figure()
            axes = mpl.pyplot.gca()
        npts = 65
        tplot = np.linspace(0, 1, npts)
        if fieldname is not None:
            cmap = mpl.pyplot.get_cmap("bwr")
        for element in self.__system._structure.elements:
            curve3D = self.getonesplinecurve(element, deformed)
            points3D = curve3D(tplot)
            if fieldname is None:
                axes.plot(points3D[:, 0], points3D[:, 1], points3D[:, 2], color="k")
            else:
                raise NotImplementedError("Field is not yet implemented")
                fieldvalues = compute_field(fieldname, element)
                axes.scatter(p[:, 0], p[:, 1], p[:, 2], cmap=cmap, c=fieldvalues)
    

def plot_mesh(mesh):
    points = mesh.points
    cells = mesh.cells
    for cell in cells:
        if "triangle" != cell.type:
            continue
        connections = cell.data
    
    mpl.pyplot.figure()
    for connection in connections:
        ps = [points[c] for c in connection]
        ps.append(ps[0])
        x = [pi[0] for pi in ps]
        y = [pi[1] for pi in ps]
        mpl.pyplot.plot(x, y, color="k")

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
    triangulation = mpl.tri.Triangulation(x, y, connections)
    mpl.pyplot.tricontourf(triangulation, v)

    

def main():
    pass
    # circle = Circle(R = 1, nu = 0.2)
    # circle = HollowCircle(Ri=0.5, Re=1.0, nu=0.2)
    circle = ThinCircle(R=1, nu=0.2)
    mesh = circle.mesh()
    show_section(function, mesh)

    mpl.pyplot.show()


if __name__ == "__main__":
    main()
