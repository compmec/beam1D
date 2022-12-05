import numpy as np
import pytest

from compmec.strct.element import EulerBernoulli
from compmec.strct.material import Isotropic
from compmec.strct.profile import Circle
from compmec.strct.system import StaticSystem


@pytest.mark.order(9)
@pytest.mark.dependency(
    depends=["tests/test_one_circle_beam_charges.py::test_end"], scope="session"
)
def test_begin():
    pass


class TestFieldCantileverCircularEulerBeam:
    def setup_beam(self):
        L = 1000
        A = (0, 0, 0)
        B = (L, 0, 0)
        self.beam = EulerBernoulli([A, B])

        profile = Circle(R=4)
        material = Isotropic(E=210e3, nu=0.3)
        self.beam.section = material, profile

    def setup_system(self):
        self.setup_beam()
        self.system = StaticSystem()
        self.system.add_element(self.beam)

        A = self.beam.path(0)
        self.system.add_BC(A, {"ux": 0, "uy": 0, "tz": 0})

    @pytest.mark.order(9)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(9)
    @pytest.mark.dependency(
        depends=["TestFieldCantileverCircularEulerBeam::test_begin"]
    )
    def test_traction(self):
        F0 = 10
        ts = np.linspace(0, 1, 11)
        self.setup_system()
        self.beam.path.knot_insert(ts)
        self.system.add_load(self.beam.path(1), {"Fx": F0})
        self.system.run()
        Utest = self.beam.field("u")

    @pytest.mark.order(9)
    @pytest.mark.dependency(
        depends=[
            "TestFieldCantileverCircularEulerBeam::test_begin",
            "TestFieldCantileverCircularEulerBeam::test_traction",
        ]
    )
    def test_end(self):
        pass


@pytest.mark.order(9)
@pytest.mark.dependency(depends=["test_begin"])
def test_end():
    pass
