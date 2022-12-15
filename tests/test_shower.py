import numpy as np
import pytest
from matplotlib import pyplot as plt

from compmec.strct.element import EulerBernoulli
from compmec.strct.material import Isotropic
from compmec.strct.profile import Circle
from compmec.strct.shower import ShowerStaticSystem
from compmec.strct.system import StaticSystem


@pytest.mark.order(9)
@pytest.mark.dependency(
    depends=[
        "tests/test_one_circle_beam_charges.py::test_end",
        "tests/test_beam_field_values.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


@pytest.mark.order(9)
@pytest.mark.dependency(depends=["test_begin"])
def test_main1():
    A = (0, 0, 0)
    B = (1000, 500, 0)
    C = (1000, -500, 0)
    D = (1000, 0, 500)
    beamAB = EulerBernoulli([A, B])
    beamAC = EulerBernoulli([A, C])
    beamAD = EulerBernoulli([A, D])
    beamBC = EulerBernoulli([B, C])
    beamBD = EulerBernoulli([B, D])
    beamCD = EulerBernoulli([C, D])
    circle = Circle(diameter=8)
    steel = Isotropic(E=210e3, nu=0.3)
    system = StaticSystem()
    for beam in [beamAB, beamAC, beamAD, beamBC, beamBD, beamCD]:
        for t in np.linspace(0, 1, 21):
            beam.path(t)
        beam.section = steel, circle
        system.add_element(beam)
    E = beamBC.path(0.5)
    system.add_BC(A, {"Ux": 0, "Uy": 0, "Uz": 0, "tx": 0, "ty": 0, "tz": 0})
    system.add_BC(B, {"Uz": 0})
    system.add_BC(C, {"Uz": 0})
    system.add_conc_load(D, {"Fz": 5000000})
    system.run()

    shower = ShowerStaticSystem(system)

    plt.figure()
    ax = plt.gca()
    shower.plot2D(projector="xy", deformed=False, axes=ax)
    shower.plot2D(projector="xy", deformed=True, axes=ax)
    plt.figure()
    plt.figure()
    ax = plt.gca()
    shower.plot2D(projector="yz", deformed=False, axes=ax)
    shower.plot2D(projector="yz", deformed=True, axes=ax)
    plt.figure()
    plt.figure()
    ax = plt.gca()
    shower.plot2D(projector="xz", deformed=False, axes=ax)
    shower.plot2D(projector="xz", deformed=True, axes=ax)
    plt.figure()
    ax = plt.axes(projection="3d")
    shower.plot3D(deformed=False, axes=ax)
    shower.plot3D(deformed=True, axes=ax)
    plt.legend()
    shower.plot2D(projector="xy", deformed=False)
    shower.plot2D(projector="xy", deformed=True)
    shower.plot3D(deformed=False)
    shower.plot3D(deformed=True)
    # plt.show()
    plt.close("all")


@pytest.mark.order(9)
@pytest.mark.dependency(depends=["test_begin"])
def test_fields():
    A = (0, 0, 0)
    B = (1000, 500, 0)
    C = (1000, -500, 0)
    D = (1000, 0, 500)
    beamAB = EulerBernoulli([A, B])
    beamAC = EulerBernoulli([A, C])
    beamAD = EulerBernoulli([A, D])
    beamBC = EulerBernoulli([B, C])
    beamBD = EulerBernoulli([B, D])
    beamCD = EulerBernoulli([C, D])
    circle = Circle(diameter=8)
    steel = Isotropic(E=210e3, nu=0.3)
    system = StaticSystem()
    for beam in [beamAB, beamAC, beamAD, beamBC, beamBD, beamCD]:
        for t in np.linspace(0, 1, 21):
            beam.path(t)
        beam.section = steel, circle
        system.add_element(beam)
    E = beamBC.path(0.5)
    system.add_BC(A, {"Ux": 0, "Uy": 0, "Uz": 0, "tx": 0, "ty": 0, "tz": 0})
    system.add_BC(B, {"Uz": 0})
    system.add_BC(C, {"Uz": 0})
    system.add_conc_load(D, {"Fz": 5000000})
    system.run()

    shower = ShowerStaticSystem(system)

    plt.figure()
    ax = plt.gca()
    shower.plot2D(projector="xy", deformed=False, fieldname="Ux", axes=ax)
    shower.plot2D(projector="xy", deformed=True, fieldname="Ux", axes=ax)
    shower.plot2D(projector="xy", deformed=False, fieldname="Uy", axes=ax)
    shower.plot2D(projector="xy", deformed=True, fieldname="Uy", axes=ax)
    shower.plot2D(projector="xy", deformed=False, fieldname="Uz", axes=ax)
    shower.plot2D(projector="xy", deformed=True, fieldname="Uz", axes=ax)
    plt.figure()
    plt.figure()
    ax = plt.gca()
    shower.plot2D(projector="yz", deformed=False, fieldname="Ux", axes=ax)
    shower.plot2D(projector="yz", deformed=True, fieldname="Ux", axes=ax)
    shower.plot2D(projector="yz", deformed=False, fieldname="Uy", axes=ax)
    shower.plot2D(projector="yz", deformed=True, fieldname="Uy", axes=ax)
    shower.plot2D(projector="yz", deformed=False, fieldname="Uz", axes=ax)
    shower.plot2D(projector="yz", deformed=True, fieldname="Uz", axes=ax)
    plt.figure()
    plt.figure()
    ax = plt.gca()
    shower.plot2D(projector="xz", deformed=False, fieldname="Ux", axes=ax)
    shower.plot2D(projector="xz", deformed=True, fieldname="Ux", axes=ax)
    shower.plot2D(projector="xz", deformed=False, fieldname="Uy", axes=ax)
    shower.plot2D(projector="xz", deformed=True, fieldname="Uy", axes=ax)
    shower.plot2D(projector="xz", deformed=False, fieldname="Uz", axes=ax)
    shower.plot2D(projector="xz", deformed=True, fieldname="Uz", axes=ax)
    plt.figure()
    ax = plt.axes(projection="3d")
    shower.plot3D(deformed=False, fieldname="Ux", axes=ax)
    shower.plot3D(deformed=True, fieldname="Ux", axes=ax)
    shower.plot3D(deformed=False, fieldname="Uy", axes=ax)
    shower.plot3D(deformed=True, fieldname="Uy", axes=ax)
    shower.plot3D(deformed=False, fieldname="Uz", axes=ax)
    shower.plot3D(deformed=True, fieldname="Uz", axes=ax)
    plt.legend()
    # plt.show()
    plt.close("all")


@pytest.mark.order(9)
@pytest.mark.dependency(depends=["test_fields"])
def test_all_axonometric():
    projectornames = [
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
    system = StaticSystem()
    beamAB = EulerBernoulli([(0, 0, 0), (10, 0, 0)])
    beamAB.section = Isotropic(E=210e3, nu=0.3), Circle(diameter=2)
    system.add_element(beamAB)
    system.add_BC((0, 0, 0), {"Ux": 0, "Uy": 0, "Uz": 0})
    system.add_conc_load((0, 0, 0), {"Fy": -10})
    with pytest.raises(ValueError):
        shower = ShowerStaticSystem(system)
    system.run()
    shower = ShowerStaticSystem(system)
    for name in projectornames:
        try:
            shower.plot2D(projector=name, deformed=False, fieldname="Ux")
        except NotImplementedError as e:
            pass


@pytest.mark.order(9)
@pytest.mark.dependency(depends=["test_fields"])
def test_all_perspective():
    projectornames = [
        "military",
        "cabinet",
        "cavalier",
        "one-point",
        "two-point",
        "three-point",
        "perspective custom",
    ]
    system = StaticSystem()
    beamAB = EulerBernoulli([(0, 0, 0), (10, 0, 0)])
    beamAB.section = Isotropic(E=210e3, nu=0.3), Circle(diameter=2)
    system.add_element(beamAB)
    system.add_BC((0, 0, 0), {"Ux": 0, "Uy": 0, "Uz": 0})
    system.add_conc_load((0, 0, 0), {"Fy": -10})
    with pytest.raises(ValueError):
        shower = ShowerStaticSystem(system)
    system.run()
    shower = ShowerStaticSystem(system)
    for name in projectornames:
        try:
            shower.plot2D(projector=name, deformed=False, fieldname="Ux")
        except NotImplementedError as e:
            pass


@pytest.mark.order(9)
@pytest.mark.dependency(depends=["test_fields"])
def test_fails():
    system = StaticSystem()
    beamAB = EulerBernoulli([(0, 0, 0), (10, 0, 0)])
    beamAB.section = Isotropic(E=210e3, nu=0.3), Circle(diameter=2)
    system.add_element(beamAB)
    system.add_BC((0, 0, 0), {"Ux": 0, "Uy": 0, "Uz": 0})
    system.add_conc_load((0, 0, 0), {"Fy": -10})
    with pytest.raises(TypeError):
        ShowerStaticSystem(1)
    with pytest.raises(TypeError):
        ShowerStaticSystem("asd")
    with pytest.raises(ValueError):
        shower = ShowerStaticSystem(system)
    system.run()
    shower = ShowerStaticSystem(system)
    with pytest.raises(TypeError):
        shower.plot2D(projector=1, deformed=False, fieldname="Ux")
    with pytest.raises(ValueError):
        shower.plot2D(projector="tt", deformed=False, fieldname="Ux")


@pytest.mark.order(9)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_main1",
        "test_fields",
        "test_all_axonometric",
        "test_all_perspective",
        "test_fails",
    ]
)
def test_end():
    pass


def main():
    test_all_axonometric()
