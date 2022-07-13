import pytest
import numpy as np
from compmec.strct.beam import EulerBernoulli
from compmec.strct.material import Isotropic
from compmec.strct.section import Circle, Square
from compmec.strct.system import StaticSystem



def test_example1():
    A = (0, 0)
    B = (1000, 0)
    beamAB = EulerBernoulli([A, B])
    beamAB.section = Circle(R=8/2, nu=0.3)
    beamAB.material = Isotropic(E=210e+3, nu=0.3)
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux":0,
                    "uy":0,
                    "tz":0})
    system.add_load(B, {"Fx": 10})
    system.run()
    Usolu = system.solution
    Ugood = np.zeros((2, 6))
    Ugood[1, 0] = 4*10*1000/(210e+3 * np.pi*8**2 )
    np.testing.assert_almost_equal(Usolu, Ugood)


def test_example2():
    A = (0, 0)
    B = (1000, 0)
    beamAB = EulerBernoulli([A, B])
    beamAB.section = Circle(R=8/2, nu=0.3)
    beamAB.material = Isotropic(E=210e+3, nu=0.3)
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux": 0,
                      "uy": 0,
                      "tz": 0})
    system.add_load(B, {"Fy": -10})
    system.run()
    Usolu = system.solution
    Ugood = np.zeros((2, 6))
    Ugood[1, 1] = -64*10*(1000**3)/(3*np.pi* 210e+3 * 8**4)
    Ugood[1, 5] = -32*10*(1000**2)/(np.pi* 210e+3 * 8**4)
    np.testing.assert_almost_equal(Usolu, Ugood)


def test_example3():
    A = (0, 0)
    B = (1000, 0)
    beamAB = EulerBernoulli([A, B])
    beamAB.section = Circle(R=8/2, nu=0.3)
    beamAB.material = Isotropic(E=210e+3, nu=0.3)
    C = beamAB.path(0.6)
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux": 0,
                      "uy": 0,
                      "tz": 0})
    system.add_load(C, {"Fy": -10})
    system.run()
    Usolu = system.solution
    Ugood = np.zeros((3, 6))
    Ugood[1, 1] = -64*10*(600**3)/(3*np.pi* 210e+3 * 8**4)
    Ugood[1, 5] = -32*10*(600**2)/(np.pi* 210e+3 * 8**4)
    Ugood[2, 5] = Ugood[1, 5]
    Ugood[2, 1] = Ugood[1, 1] + 400 * Ugood[1, 5]
    np.testing.assert_almost_equal(Usolu, Ugood)


def test_example4():
    A = (0, 0)
    B = (1000, 0)
    beamAB = EulerBernoulli([A, B])
    beamAB.section = Circle(R=8/2, nu=0.3)
    beamAB.material = Isotropic(E=210e+3, nu=0.3)
    C = beamAB.path(0.3)
    D = beamAB.path(0.7)
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux": 0,
                      "uy": 0,
                      "tz": 0})
    system.add_load(C, {"Fy": -10})
    system.add_load(D, {"Fy": -25})
    system.run()


def test_example5():
    A = (0, 0)
    B = (1000, 0)
    beamAB = EulerBernoulli([A, B])
    beamAB.section = Circle(R=8/2, nu=0.3)
    beamAB.material = Isotropic(E=210e+3, nu=0.3)
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux": 0,
                      "uy": 0,
                      "tz": 0})
    system.add_dist_load(beamAB, (0, 1), {"Fy": (-10, -10)})
    system.run()

def test_example6():
    A = (0, 0)
    B = (1000, 0)
    beamAB = EulerBernoulli([A, B])
    beamAB.section = Circle(R=8/2, nu=0.3)
    beamAB.material = Isotropic(E=210e+3, nu=0.3)
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux": 0,
                      "uy": 0,
                      "tz": 0})
    system.add_dist_load(beamAB, (0.3, 0.7), {"Fy": (-10, -10)})
    system.run()

def test_example7():
    A = (0, 0)
    B = (1000, 0)
    beamAB = EulerBernoulli([A, B])
    beamAB.section = Circle(R=8/2, nu=0.3)
    beamAB.material = Isotropic(E=210e+3, nu=0.3)
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux": 0,
                      "uy": 0,
                      "tz": 0})
    system.add_dist_load(beamAB, (0, 1), {"Fy": (-10, 0)})
    system.run()

def test_example8():
    A = (0, 0)
    B = (1000, 0)
    beamAB = EulerBernoulli([A, B])
    beamAB.section = Circle(R=8/2, nu=0.3)
    beamAB.material = Isotropic(E=210e+3, nu=0.3)
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux": 0,
                      "uy": 0,
                      "tz": 0})
    system.add_dist_load(beamAB, (0, 0.3, 0.7, 1), {"Fy": (0, -10, -10, 0)})
    system.run()

def test_example9():
    A = (0, 0)
    B = (1000, 0)
    C = (500, 500)
    beamAB = EulerBernoulli([A, B])
    beamAC = EulerBernoulli([A, C])
    beamBC = EulerBernoulli([B, C])
    circle = Circle(R=8/2, nu=0.3)
    square = Square(b=8, nu=0.3)
    steel = Isotropic(E=210e+3, nu=0.3)
    beamAB.section = square
    beamBC.section = circle
    beamAC.section = circle
    for beam in [beamAB, beamBC, beamAC]:
        beam.material = steel
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux": 0,
                      "uy": 0})
    system.add_BC(B, {"uy": 0})
    system.add_load(C, {"Fx": 15,
                        "Fy": -10})
    system.run()
    

if __name__ == "__main__":
    # pytest.main()
    test_example5()
