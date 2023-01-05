from typing import Tuple

import numpy as np
import pytest
from matplotlib import pyplot as plt

from compmec.strct.element import EulerBernoulli
from compmec.strct.material import Isotropic
from compmec.strct.profile import Circle, Retangular
from compmec.strct.shower import ShowerStaticSystem
from compmec.strct.system import StaticSystem


@pytest.mark.order(10)
@pytest.mark.dependency(
    depends=[
        "tests/test_one_circle_truss_charges.py::test_end",
        "tests/test_one_circle_beam_charges.py::test_end",
        # "tests/test_one_retangular_beam_charges.py::test_end",
        "tests/test_beam_field_values.py::test_end",
        "tests/test_shower.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


@pytest.mark.order(10)
@pytest.mark.timeout(15)
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
    system.add_BC(A, "Ux", 0)
    system.add_BC(A, "Uy", 0)
    system.add_BC(A, "tz", 0)
    system.add_conc_load(B, "Fx", 10)
    system.run()

    tsample = np.linspace(0, 1, 129)
    displacement = beamAB.field("U")(tsample)
    plt.close("all")
    plt.figure()
    plt.plot(tsample, displacement[:, 0], label="Ux")
    plt.plot(tsample, displacement[:, 1], label="Uy")
    plt.plot(tsample, displacement[:, 2], label="Uz")
    plt.legend()

    shower = ShowerStaticSystem(system)
    plt.figure()
    ax = plt.gca()
    shower.plot2D(projector="xy", deformed=False, axes=ax)
    shower.plot2D(projector="xy", deformed=True, axes=ax)
    plt.figure()
    ax = plt.axes(projection="3d")
    shower.plot3D(deformed=False, axes=ax)
    shower.plot3D(deformed=True, axes=ax)
    plt.legend()


@pytest.mark.order(10)
@pytest.mark.timeout(10)
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
    system.add_BC(A, "Ux", 0)
    system.add_BC(A, "Uy", 0)
    system.add_BC(A, "tz", 0)
    system.add_conc_load(B, "Fy", 10)
    system.run()

    tsample = np.linspace(0, 1, 129)
    displacement = beamAB.field("U")(tsample)
    plt.close("all")
    plt.figure()
    plt.plot(tsample, displacement[:, 0], label="Ux")
    plt.plot(tsample, displacement[:, 1], label="Uy")
    plt.plot(tsample, displacement[:, 2], label="Uz")
    plt.legend()

    shower = ShowerStaticSystem(system)
    plt.figure()
    ax = plt.gca()
    shower.plot2D(projector="xy", deformed=False, axes=ax)
    shower.plot2D(projector="xy", deformed=True, axes=ax)
    plt.figure()
    ax = plt.axes(projection="3d")
    shower.plot3D(deformed=False, axes=ax)
    shower.plot3D(deformed=True, axes=ax)
    plt.legend()


@pytest.mark.order(10)
@pytest.mark.timeout(10)
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
    system.add_BC(A, "Ux", 0)
    system.add_BC(A, "Uy", 0)
    system.add_BC(A, "tz", 0)
    system.add_conc_load(C, "Fy", -10)
    system.run()

    tsample = np.linspace(0, 1, 129)
    displacement = beamAB.field("U")(tsample)
    plt.close("all")
    plt.figure()
    plt.plot(tsample, displacement[:, 0], label="Ux")
    plt.plot(tsample, displacement[:, 1], label="Uy")
    plt.plot(tsample, displacement[:, 2], label="Uz")
    plt.legend()

    shower = ShowerStaticSystem(system)
    plt.figure()
    ax = plt.gca()
    shower.plot2D(projector="xy", deformed=False, axes=ax)
    shower.plot2D(projector="xy", deformed=True, axes=ax)
    plt.figure()
    ax = plt.axes(projection="3d")
    shower.plot3D(deformed=False, axes=ax)
    shower.plot3D(deformed=True, axes=ax)
    plt.legend()


@pytest.mark.order(10)
@pytest.mark.timeout(10)
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
    system.add_BC(A, "Ux", 0)
    system.add_BC(A, "Uy", 0)
    system.add_BC(A, "tz", 0)
    system.add_conc_load(C, "Fy", -12)
    system.add_conc_load(D, "Fy", -24)
    system.run()

    tsample = np.linspace(0, 1, 129)
    displacement = beamAB.field("U")(tsample)
    plt.close("all")
    plt.figure()
    plt.plot(tsample, displacement[:, 0], label="Ux")
    plt.plot(tsample, displacement[:, 1], label="Uy")
    plt.plot(tsample, displacement[:, 2], label="Uz")
    plt.legend()

    shower = ShowerStaticSystem(system)
    plt.figure()
    ax = plt.gca()
    shower.plot2D(projector="xy", deformed=False, axes=ax)
    shower.plot2D(projector="xy", deformed=True, axes=ax)
    plt.figure()
    ax = plt.axes(projection="3d")
    shower.plot3D(deformed=False, axes=ax)
    shower.plot3D(deformed=True, axes=ax)
    plt.legend()


@pytest.mark.order(10)
@pytest.mark.timeout(10)
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
    system.add_BC(A, "Ux", 0)
    system.add_BC(A, "Uy", 0)
    system.add_BC(A, "tz", 0)

    def distributed_load(t: float) -> float:
        return q0

    system.add_dist_load(beamAB, "Fy", distributed_load)
    system.run()

    tsample = np.linspace(0, 1, 129)
    displacement = beamAB.field("U")(tsample)
    plt.close("all")
    plt.figure()
    plt.plot(tsample, displacement[:, 0], label="Ux")
    plt.plot(tsample, displacement[:, 1], label="Uy")
    plt.plot(tsample, displacement[:, 2], label="Uz")
    plt.legend()

    shower = ShowerStaticSystem(system)
    plt.figure()
    ax = plt.gca()
    shower.plot2D(projector="xy", deformed=False, axes=ax)
    shower.plot2D(projector="xy", deformed=True, axes=ax)
    plt.figure()
    ax = plt.axes(projection="3d")
    shower.plot3D(deformed=False, axes=ax)
    shower.plot3D(deformed=True, axes=ax)
    plt.legend()


@pytest.mark.order(10)
@pytest.mark.timeout(5)
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
    system.add_BC(A, "Ux", 0)
    system.add_BC(A, "Uy", 0)
    system.add_BC(A, "tz", 0)

    def distributed_load(t: float) -> float:
        if t < 0.3:
            return 0
        if 0.3 <= t <= 0.7:
            return -10
        return 0

    system.add_dist_load(beamAB, "Fy", distributed_load)
    system.run()

    tsample = np.linspace(0, 1, 129)
    displacement = beamAB.field("U")(tsample)
    plt.close("all")
    plt.figure()
    plt.plot(tsample, displacement[:, 0], label="Ux")
    plt.plot(tsample, displacement[:, 1], label="Uy")
    plt.plot(tsample, displacement[:, 2], label="Uz")
    plt.legend()

    shower = ShowerStaticSystem(system)
    plt.figure()
    ax = plt.gca()
    shower.plot2D(projector="xy", deformed=False, axes=ax)
    shower.plot2D(projector="xy", deformed=True, axes=ax)
    plt.figure()
    ax = plt.axes(projection="3d")
    shower.plot3D(deformed=False, axes=ax)
    shower.plot3D(deformed=True, axes=ax)
    plt.legend()


@pytest.mark.order(10)
@pytest.mark.timeout(10)
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
    system.add_BC(A, "Ux", 0)
    system.add_BC(A, "Uy", 0)
    system.add_BC(A, "tz", 0)

    def distributed_load(t: float) -> Tuple[float, float, float]:
        return -10 * (1 - t)

    system.add_dist_load(beamAB, "Fy", distributed_load)
    system.run()

    tsample = np.linspace(0, 1, 129)
    displacement = beamAB.field("U")(tsample)
    plt.close("all")
    plt.figure()
    plt.plot(tsample, displacement[:, 0], label="Ux")
    plt.plot(tsample, displacement[:, 1], label="Uy")
    plt.plot(tsample, displacement[:, 2], label="Uz")
    plt.legend()

    shower = ShowerStaticSystem(system)
    plt.figure()
    ax = plt.gca()
    shower.plot2D(projector="xy", deformed=False, axes=ax)
    shower.plot2D(projector="xy", deformed=True, axes=ax)
    plt.figure()
    ax = plt.axes(projection="3d")
    shower.plot3D(deformed=False, axes=ax)
    shower.plot3D(deformed=True, axes=ax)
    plt.legend()


@pytest.mark.order(10)
@pytest.mark.timeout(10)
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
    system.add_BC(A, "Ux", 0)
    system.add_BC(A, "Uy", 0)
    system.add_BC(A, "tz", 0)

    def distributed_load(t: float) -> Tuple[float, float, float]:
        if 0.3 <= t <= 0.7:
            return -10
        elif 0.0 <= t <= 0.3:
            return -10 * t / 0.3
        elif 0.7 <= t <= 1:
            return -10 * (1 - t) / 0.3

    system.add_dist_load(beamAB, "Fy", distributed_load)
    system.run()

    tsample = np.linspace(0, 1, 129)
    displacement = beamAB.field("U")(tsample)
    plt.close("all")
    plt.figure()
    plt.plot(tsample, displacement[:, 0], label="Ux")
    plt.plot(tsample, displacement[:, 1], label="Uy")
    plt.plot(tsample, displacement[:, 2], label="Uz")
    plt.legend()

    shower = ShowerStaticSystem(system)
    plt.figure()
    ax = plt.gca()
    shower.plot2D(projector="xy", deformed=False, axes=ax)
    shower.plot2D(projector="xy", deformed=True, axes=ax)
    plt.figure()
    ax = plt.axes(projection="3d")
    shower.plot3D(deformed=False, axes=ax)
    shower.plot3D(deformed=True, axes=ax)
    plt.legend()


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
    system.add_BC(A, "Ux", 0)
    system.add_BC(A, "Uy", 0)
    system.add_BC(B, "Uy", 0)
    system.add_conc_load(C, "Fx", 1500000)
    system.add_conc_load(C, "Fy", -1000000)
    system.run()

    tsample = np.linspace(0, 1, 129)
    displacement = beamAB.field("U")(tsample)
    plt.close("all")
    plt.figure()
    plt.plot(tsample, displacement[:, 0], label="Ux")
    plt.plot(tsample, displacement[:, 1], label="Uy")
    plt.plot(tsample, displacement[:, 2], label="Uz")
    plt.legend()

    shower = ShowerStaticSystem(system)
    plt.figure()
    ax = plt.gca()
    shower.plot2D(projector="xy", deformed=False, axes=ax)
    shower.plot2D(projector="xy", deformed=True, axes=ax)
    plt.figure()
    ax = plt.axes(projection="3d")
    shower.plot3D(deformed=False, axes=ax)
    shower.plot3D(deformed=True, axes=ax)
    plt.legend()


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
    system.add_BC(A, "Ux", 0)
    system.add_BC(A, "Uy", 0)
    system.add_BC(A, "tz", 0)
    system.add_BC(B, "Ux", 0)
    system.add_BC(B, "Uy", 0)
    system.add_BC(B, "tz", 0)

    def distributed_load_beamBC(t: float) -> Tuple[float, float, float]:
        return -0.1

    def distributed_load_beamAC(t: float) -> Tuple[float, float, float]:
        return -0.1

    system.add_dist_load(beamBC, "Fy", distributed_load_beamBC)
    system.add_dist_load(beamAC, "Fx", distributed_load_beamAC)
    system.run()

    tsample = np.linspace(0, 1, 129)
    displacement = beamAC.field("U")(tsample)
    plt.close("all")
    plt.figure()
    plt.plot(tsample, displacement[:, 0], label="Ux")
    plt.plot(tsample, displacement[:, 1], label="Uy")
    plt.plot(tsample, displacement[:, 2], label="Uz")
    plt.legend()

    shower = ShowerStaticSystem(system)
    plt.figure()
    ax = plt.gca()
    shower.plot2D(projector="xy", deformed=False, axes=ax)
    shower.plot2D(projector="xy", deformed=True, axes=ax)
    plt.figure()
    ax = plt.axes(projection="3d")
    shower.plot3D(deformed=False, axes=ax)
    shower.plot3D(deformed=True, axes=ax)
    plt.legend()


@pytest.mark.order(10)
@pytest.mark.dependency(depends=["test_begin", "test_example10"])
def test_end():
    pass
