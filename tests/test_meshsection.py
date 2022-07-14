from compmec.strct.section import Circle, HollowCircle, Retangular, ThinCircle
from matplotlib import pyplot as plt


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

def plot_rectangle():
    retangle = Retangular(b=1, h=2, nu=0.2)
    mesh = retangle.mesh()
    plot_mesh(mesh)

def plot_circle():
    circle = Circle(R=1, nu=0.2)
    mesh = circle.mesh()
    plot_mesh(mesh)
    

def plot_hollowcircle():
    circle = HollowCircle(Ri=0.5, Re=1.0, nu=0.2)
    mesh = circle.mesh()
    plot_mesh(mesh)

def plot_thincircle():
    circle = ThinCircle(R=1.0, nu=0.2)
    mesh = circle.mesh()
    plot_mesh(mesh)

if __name__ == "__main__":
    plot_circle()
    plot_hollowcircle()
    plot_thincircle()
    # plot_rectangle()
    plt.show()