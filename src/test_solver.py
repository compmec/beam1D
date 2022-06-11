import numpy as np
from numpy import linalg as la
from material import Isotropic
from beam import EulerBernoulli
from section import Circle
from solver import solve
from matplotlib import pyplot as plt


def find_index(p : np.ndarray, points : np.ndarray) -> int:
    npts = points.shape[0]
    diff = points[:] - p
    
    distsquare = np.array([ np.sum(diff[i]**2) for i in range(npts)])
    ind = np.where(distsquare == np.min(distsquare))[0][0] 
    return ind

def position_index(position : str) -> int:
    if position == "ux":
        return 0
    if position == "uy":
        return 1
    if position == "uz":
        return 2
    if position == "tx":
        return 3
    if position == "ty":
        return 4
    if position == "tz":
        return 5
    raise Exception("Not expected")

def force_index(force : str) -> int:
    if force == "Fx":
        return 0
    if force == "Fy":
        return 1
    if force == "Fz":
        return 2
    if force == "Mx":
        return 3
    if force == "My":
        return 4
    if force == "Mz":
        return 5
    raise Exception("Not expected")

def example1():
    

    A = (0, 0, 0)  # mm
    B = (1000, 0, 0)  # mm
    C = (500, 500, 0)  # mm
    E = 90000 # MPa
    nu = 0.45  # Poisson coeficient
    steel = Isotropic(E=E, nu=nu)
    circle = Circle(nu=nu, R=1)

    points = np.array([A, B, C])
    beams = []
    beams.append(EulerBernoulli(A, B))
    beams.append(EulerBernoulli(A, C))
    beams.append(EulerBernoulli(B, C))
    for beam in beams:
        beam.material = steel
        beam.section = circle

    u_known = [(A, {"ux": 0, "uy":0}),
              (B, {"uy": 0})]
    forces = [(C, {"Fx": 100000, "Fy": 1}),
              (B, {"Fx": 30000})]


    npts = len(points)
    Kexp = np.zeros((npts, 6, npts, 6), dtype="float64")
    for beam in beams:
        ind0 = find_index(beam.p0, points)
        ind1 = find_index(beam.p1, points)
        K = beam.stiffness_matrix()
        Kexp[ind0, :, ind0, :] += K[0, :, 0, :]
        Kexp[ind0, :, ind1, :] += K[0, :, 1, :]
        Kexp[ind1, :, ind0, :] += K[1, :, 0, :]
        Kexp[ind1, :, ind1, :] += K[1, :, 1, :]
        
    U = np.empty((npts, 6), dtype="object")
    F = np.zeros((npts, 6))

    for point, dict_knownval in u_known:
        index_point = find_index(point, points)
        for key, val in dict_knownval.items():
            index_position = position_index(key)
            U[index_point, index_position] = val

    for point, dict_knownval in forces:
        index_point = find_index(point, points)
        for key, val in dict_knownval.items():
            index_position = force_index(key)
            F[index_point, index_position] = val

    print("Before")
    print("U = ")
    print(U)
    print("F = ")
    print(F)

    U, F = solve(Kexp, F, U)
    print("After")
    print("U = ")
    print(U)
    print("F = ")
    print(F)

    new_points = points + U[:, :3]
    for beam in beams:
        ind0 = find_index(beam.p0, points)
        ind1 = find_index(beam.p1, points)
        x = (beam.p0[0], beam.p1[0])
        y = (beam.p0[1], beam.p1[1])
        plt.plot(x, y, color="blue")
        x = (new_points[ind0][0], new_points[ind1][0])
        y = (new_points[ind0][1], new_points[ind1][1])
        plt.plot(x, y, color="red")
    plt.show()
    

def example2():
    

    A = (0, 0, 0)  # mm
    B = (1000, 0, 0)  # mm
    C = (0, 1000, 0)  # mm
    D = (1000, 1000, 0)  # mm
    E = 90000 # MPa
    nu = 0.45  # Poisson coeficient
    steel = Isotropic(E=E, nu=nu)
    circle = Circle(nu=nu, R=1)

    points = np.array([A, B, C, D])
    beams = []
    beams.append(EulerBernoulli(A, B))
    beams.append(EulerBernoulli(A, C))
    beams.append(EulerBernoulli(A, D))
    beams.append(EulerBernoulli(B, D))
    beams.append(EulerBernoulli(B, C))
    beams.append(EulerBernoulli(C, D))
    for beam in beams:
        beam.material = steel
        beam.section = circle

    u_known = [(A, {"ux": 0, "uy":0}),
              (B, {"uy": 0})]
    forces = [(C, {"Fy": 1}),
              (B, {"Fx": 30000})]


    npts = len(points)
    Kexp = np.zeros((npts, 6, npts, 6), dtype="float64")
    for beam in beams:
        ind0 = find_index(beam.p0, points)
        ind1 = find_index(beam.p1, points)
        K = beam.stiffness_matrix()
        Kexp[ind0, :, ind0, :] += K[0, :, 0, :]
        Kexp[ind0, :, ind1, :] += K[0, :, 1, :]
        Kexp[ind1, :, ind0, :] += K[1, :, 0, :]
        Kexp[ind1, :, ind1, :] += K[1, :, 1, :]
        
    U = np.empty((npts, 6), dtype="object")
    F = np.zeros((npts, 6))

    for point, dict_knownval in u_known:
        index_point = find_index(point, points)
        for key, val in dict_knownval.items():
            index_position = position_index(key)
            U[index_point, index_position] = val

    for point, dict_knownval in forces:
        index_point = find_index(point, points)
        for key, val in dict_knownval.items():
            index_position = force_index(key)
            F[index_point, index_position] = val

    print("Before")
    print("U = ")
    print(U)
    print("F = ")
    print(F)

    U, F = solve(Kexp, F, U)
    print("After")
    print("U = ")
    print(U)
    print("F = ")
    print(F)

    new_points = points + U[:, :3]
    for beam in beams:
        ind0 = find_index(beam.p0, points)
        ind1 = find_index(beam.p1, points)
        x = (beam.p0[0], beam.p1[0])
        y = (beam.p0[1], beam.p1[1])
        plt.plot(x, y, color="blue")
        x = (new_points[ind0][0], new_points[ind1][0])
        y = (new_points[ind0][1], new_points[ind1][1])
        plt.plot(x, y, color="red")
    plt.show()

if __name__ == "__main__":
    np.set_printoptions(precision=2)
    example1()
    # example2()