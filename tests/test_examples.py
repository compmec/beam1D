import numpy as np
import pytest
from matplotlib import pyplot as plt

from compmec.strct.element import EulerBernoulli
from compmec.strct.material import Isotropic
from compmec.strct.profile import Circle, Retangular
from compmec.strct.system import StaticSystem


@pytest.mark.order(10)
@pytest.mark.dependency(
    depends=[
        "tests/test_one_circle_beam_charges.py::test_end",
        # "tests/test_one_retangular_beam_charges.py::test_end",
        "tests/test_beam_field_values.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


@pytest.mark.order(10)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_example1():
    A = (0, 0, 0)
    B = (1000, 0, 0)
    beamAB = EulerBernoulli([A, B])
    steel = Isotropic(E=210e3, nu=0.3)
    circle = Circle(diameter=8)
    beamAB.section = steel, circle
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux": 0, "uy": 0, "tz": 0})
    system.add_load(B, {"Fx": 10})
    system.run()

    displacement = beamAB.field("u")


@pytest.mark.order(10)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_example1"])
def test_example2():
    A = (0, 0, 0)
    B = (1000, 0, 0)
    beamAB = EulerBernoulli([A, B])
    steel = Isotropic(E=210e3, nu=0.3)
    circle = Circle(diameter=8)
    beamAB.section = steel, circle
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux": 0, "uy": 0, "tz": 0})
    system.add_load(B, {"Fy": -10})
    system.run()
    Usolu = system.solution
    Ugood = np.zeros((2, 6))
    Ugood[1, 1] = -64 * 10 * (1000**3) / (3 * np.pi * 210e3 * 8**4)
    Ugood[1, 5] = -32 * 10 * (1000**2) / (np.pi * 210e3 * 8**4)
    np.testing.assert_almost_equal(Usolu, Ugood)


@pytest.mark.order(10)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_example2"])
def test_example3():
    A = (0, 0, 0)
    B = (1000, 0, 0)
    beamAB = EulerBernoulli([A, B])
    steel = Isotropic(E=210e3, nu=0.3)
    circle = Circle(diameter=8)
    beamAB.section = steel, circle
    C = beamAB.path(0.6)
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux": 0, "uy": 0, "tz": 0})
    system.add_load(C, {"Fy": -10})
    system.run()
    Usolu = system.solution
    Ugood = np.zeros((3, 6))
    Ugood[1, 1] = -64 * 10 * (600**3) / (3 * np.pi * 210e3 * 8**4)
    Ugood[1, 5] = -32 * 10 * (600**2) / (np.pi * 210e3 * 8**4)
    Ugood[2, 5] = Ugood[1, 5]
    Ugood[2, 1] = Ugood[1, 1] + 400 * Ugood[1, 5]
    np.testing.assert_almost_equal(Usolu, Ugood)


@pytest.mark.order(10)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_example3"])
def test_example4():
    A = (0, 0, 0)
    B = (1000, 0, 0)
    beamAB = EulerBernoulli([A, B])
    steel = Isotropic(E=210e3, nu=0.3)
    circle = Circle(diameter=8)
    beamAB.section = steel, circle
    C = beamAB.path(0.3)
    D = beamAB.path(0.7)
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux": 0, "uy": 0, "tz": 0})
    system.add_load(C, {"Fy": -12})
    system.add_load(D, {"Fy": -24})
    system.run()


@pytest.mark.order(10)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_example4"])
def test_example5():
    q0 = -0.1
    L = 1000
    EI = 210e3 * np.pi * (8**4) / 64
    A = (0, 0, 0)
    B = (1000, 0, 0)
    beamAB = EulerBernoulli([A, B])
    steel = Isotropic(E=210e3, nu=0.3)
    circle = Circle(diameter=8)
    beamAB.section = steel, circle
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux": 0, "uy": 0, "tz": 0})
    system.add_dist_load(beamAB, (0, 1), {"Fy": (q0, q0)})
    system.run()


@pytest.mark.order(10)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_example5"])
def test_example6():
    A = (0, 0, 0)
    B = (1000, 0, 0)
    beamAB = EulerBernoulli([A, B])
    steel = Isotropic(E=210e3, nu=0.3)
    circle = Circle(diameter=8)
    beamAB.section = steel, circle
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux": 0, "uy": 0, "tz": 0})
    system.add_dist_load(beamAB, (0.3, 0.7), {"Fy": (-10, -10)})
    system.run()


@pytest.mark.order(10)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_example6"])
def test_example7():
    A = (0, 0, 0)
    B = (1000, 0, 0)
    beamAB = EulerBernoulli([A, B])
    steel = Isotropic(E=210e3, nu=0.3)
    circle = Circle(diameter=8)
    beamAB.section = steel, circle
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux": 0, "uy": 0, "tz": 0})
    system.add_dist_load(beamAB, (0, 1), {"Fy": (-10, 0)})
    system.run()


@pytest.mark.order(10)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_example7"])
def test_example8():
    A = (0, 0, 0)
    B = (1000, 0, 0)
    beamAB = EulerBernoulli([A, B])
    steel = Isotropic(E=210e3, nu=0.3)
    circle = Circle(diameter=8)
    beamAB.section = steel, circle
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux": 0, "uy": 0, "tz": 0})
    system.add_dist_load(beamAB, (0, 0.3, 0.7, 1), {"Fy": (0, -10, -10, 0)})
    system.run()


@pytest.mark.order(10)
@pytest.mark.timeout(6)
@pytest.mark.dependency(depends=["test_example8"])
def test_example9():
    A = (0, 0, 0)
    B = (1000, 0, 0)
    C = (500, 500, 0)
    beamAB = EulerBernoulli([A, B])
    beamAC = EulerBernoulli([A, C])
    beamBC = EulerBernoulli([B, C])
    circle = Circle(diameter=8)
    square = Retangular(8, 8)
    steel = Isotropic(E=210e3, nu=0.3)
    beamAB.section = steel, square
    beamBC.section = steel, circle
    beamAC.section = steel, circle
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_element(beamBC)
    system.add_element(beamAC)
    system.add_BC(A, {"ux": 0, "uy": 0})
    system.add_BC(B, {"uy": 0})
    system.add_load(C, {"Fx": 1500000, "Fy": -1000000})
    system.run()


@pytest.mark.order(10)
@pytest.mark.timeout(6)
@pytest.mark.dependency(depends=["test_example9"])
def test_example10():
    A = (300, 0, 0)
    B = (0, 500, 0)
    C = (300, 500, 0)
    beamAC = EulerBernoulli([A, C])
    beamBC = EulerBernoulli([B, C])
    circle = Circle(diameter=8)
    steel = Isotropic(E=210e3, nu=0.3)
    beamAC.section = steel, circle
    beamBC.section = steel, circle
    system = StaticSystem()
    system.add_element(beamAC)
    system.add_element(beamBC)
    system.add_BC(A, {"ux": 0, "uy": 0, "tz": 0})
    system.add_BC(B, {"ux": 0, "uy": 0, "tz": 0})
    system.add_dist_load(beamBC, (0, 1), {"Fy": (-0.1, -0.1)})
    system.add_dist_load(beamAC, (0, 1), {"Fx": (-0.1, -0.1)})
    system.run()


@pytest.mark.order(10)
@pytest.mark.dependency(depends=["test_begin", "test_example10"])
def test_end():
    pass
