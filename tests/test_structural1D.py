import numpy as np
import pytest
from compmec.nurbs import GeneratorKnotVector, SplineCurve

from compmec.strct.element import EulerBernoulli, Structural1D
from compmec.strct.material import Isotropic
from compmec.strct.profile import Circle
from compmec.strct.section import CircleSection


@pytest.mark.order(5)
@pytest.mark.dependency(
    depends=[
        "tests/test_material.py::test_end",
        "tests/test_geometry.py::test_end",
        "tests/test_section.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


@pytest.mark.order(5)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_Structural1Dlinearpath():
    ntests = 10
    for i in range(ntests):
        PointA = np.random.uniform(-1, 1, 3)
        PointB = np.random.uniform(-1, 1, 3)
        tsample = np.linspace(0, 1, 10)
        structural = Structural1D([PointA, PointB])
        for ti in tsample:
            pgood = (1 - ti) * PointA + ti * PointB
            ptest = structural.path(ti)
            np.testing.assert_allclose(ptest, pgood)


class InitBeam(object):
    def create_random_isotropic_material(self):
        E = np.random.uniform(100, 200)
        nu = np.random.uniform(0.01, 0.49)
        self.material = Isotropic(E=E, nu=nu)

    def create_random_circle_profile(self):
        R = np.random.uniform(1, 2)
        self.profile = Circle(radius=R)

    def create_random_circle_section(self):
        self.create_random_circle_profile()
        self.create_random_isotropic_material()
        self.section = CircleSection(self.material, self.profile)


class TestEulerBernoulli(InitBeam):
    def create_beam(self):
        A = (0, 0, 0)
        B = (1, 0, 0)
        path = [A, B]
        self.beam = EulerBernoulli(path)

    @pytest.mark.order(5)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(5)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(depends=["TestEulerBernoulli::test_begin"])
    def test_creation(self):
        self.create_beam()

    @pytest.mark.order(5)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(depends=["TestEulerBernoulli::test_creation"])
    def test_set_section(self):
        self.create_random_circle_section()
        self.create_beam()
        self.beam.section = self.section

    @pytest.mark.order(5)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(depends=["TestEulerBernoulli::test_creation"])
    def test_set_tuple_material_profile(self):
        self.create_random_isotropic_material()
        self.create_random_circle_profile()
        self.create_beam()
        material = self.material
        profile = self.profile
        self.beam.section = material, profile

    @pytest.mark.order(5)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(depends=["TestEulerBernoulli::test_creation"])
    def test_fail_creation_class(self):
        A = [[0, 0, 0]]
        B = [[0, 0, 1]]
        with pytest.raises(TypeError):
            self.beam = EulerBernoulli([A, B])
        A = [(0, 0, 0), (1, 2, 3)]
        B = [(3, 4, 5), (6, 7, 8)]
        with pytest.raises(TypeError):
            self.beam = EulerBernoulli([A, B])
        A = [0, 0, 0, 0]
        B = [1, 0, 0, 0]
        with pytest.raises(ValueError):
            self.beam = EulerBernoulli([A, B])

        with pytest.raises(TypeError):
            self.beam = EulerBernoulli("asd")
        with pytest.raises(TypeError):
            self.beam = EulerBernoulli(1)

    @pytest.mark.order(5)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(depends=["TestEulerBernoulli::test_creation"])
    def test_fail_set_section(self):
        self.create_random_isotropic_material()
        self.create_random_circle_profile()
        self.create_beam()
        material = self.material
        profile = self.profile
        with pytest.raises(TypeError):
            self.beam.section = material
        with pytest.raises(TypeError):
            self.beam.section = profile
        with pytest.raises(TypeError):
            self.beam.section = 1, 1
        with pytest.raises(TypeError):
            self.beam.section = material, 1
        with pytest.raises(TypeError):
            self.beam.section = 1, profile
        with pytest.raises(ValueError):
            self.beam.section = material, profile, 1

    @pytest.mark.order(5)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(depends=["TestEulerBernoulli::test_creation"])
    def test_set_from_curve(self):
        self.create_random_isotropic_material()
        self.create_random_circle_profile()
        knotvector = GeneratorKnotVector.uniform(2, 5)
        ctrlpoints = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1)]
        curve = SplineCurve(knotvector, ctrlpoints)
        self.beam = EulerBernoulli(curve)

    @pytest.mark.order(5)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestEulerBernoulli::test_creation",
            "TestEulerBernoulli::test_set_section",
            "TestEulerBernoulli::test_set_tuple_material_profile",
            "TestEulerBernoulli::test_fail_creation_class",
            "TestEulerBernoulli::test_fail_set_section",
            "TestEulerBernoulli::test_set_from_curve",
        ]
    )
    def test_end(self):
        pass


@pytest.mark.order(5)
@pytest.mark.timeout(2)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_Structural1Dlinearpath",
        "TestEulerBernoulli::test_end",
    ]
)
def test_end():
    pass
