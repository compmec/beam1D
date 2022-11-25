import numpy as np
import pytest

from compmec.strct.element import EulerBernoulli, Structural1D
from compmec.strct.material import Isotropic
from compmec.strct.profile import Circle
from compmec.strct.section import CircleSection


@pytest.mark.order(5)
@pytest.mark.dependency()
def test_begin():
    pass


@pytest.mark.order(5)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_Structural1Dlinearpath():
    ntests = 10
    for i in range(ntests):
        p0 = np.random.rand(3)
        p1 = np.random.rand(3)
        t = np.linspace(0, 1, 10)
        structural = Structural1D([p0, p1])
        for ti in t:
            pgood = (1 - ti) * p0 + ti * p1
            ptest = structural.path(ti)
            np.testing.assert_allclose(ptest, pgood)


class InitBeam(object):
    def create_random_isotropic_material(self):
        E = np.random.uniform(100, 200)
        nu = np.random.uniform(0.01, 0.49)
        self.material = Isotropic(E=E, nu=nu)

    def create_random_circle_profile(self):
        R = np.random.uniform(1, 2)
        self.profile = Circle(R=R)

    def create_random_circle_section(self):
        self.create_random_circle_profile()
        self.create_random_isotropic_material()
        self.section = CircleSection(self.material, self.profile)


@pytest.mark.order(5)
@pytest.mark.timeout(2)
@pytest.mark.dependency()
class TestEulerBernoulli(InitBeam):
    def create_beam(self):
        A = (0, 0, 0)
        B = (1, 0, 0)
        path = [A, B]
        self.beam = EulerBernoulli(path)

    @pytest.mark.dependency()
    def test_creation(self):
        self.create_beam()

    @pytest.mark.dependency(depends=["TestEulerBernoulli::test_creation"])
    def test_set_section(self):
        self.create_random_circle_section()
        self.create_beam()
        self.beam.section = self.section

    @pytest.mark.dependency(depends=["TestEulerBernoulli::test_creation"])
    def test_set_tuple_material_profile(self):
        self.create_random_isotropic_material()
        self.create_random_circle_profile()
        self.create_beam()
        material = self.material
        profile = self.profile
        self.beam.section = material, profile


@pytest.mark.order(5)
# @pytest.mark.skip(reason="Not test now")
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin", "test_Structural1Dlinearpath"])
def test_end():
    pass
