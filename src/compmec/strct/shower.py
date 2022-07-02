from matplotlib import tri
from matplotlib import pyplot as plt
from compmec.strct.section import Circle, HollowCircle, ThinCircle
import numpy as np


class Shower(object):
    def __init__(self):
        pass

    

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
