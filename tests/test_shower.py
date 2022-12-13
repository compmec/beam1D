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
def test_example20():
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
    system.add_BC(A, {"ux": 0, "uy": 0, "uz": 0, "tx": 0, "ty": 0, "tz": 0})
    system.add_BC(B, {"uz": 0})
    system.add_BC(C, {"uz": 0})
    system.add_load(D, {"Fz": 5000000})
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
    # plt.show()
    plt.close("all")


@pytest.mark.order(9)
@pytest.mark.dependency(depends=["test_begin", "test_example20"])
def test_end():
    pass
