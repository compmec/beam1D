import numpy as np
import pytest

from compmec.strct.material import Isotropic
from compmec.strct.profile import Circle, HollowCircle, Retangular, ThinCircle
from compmec.strct.section import CircleSection, HollowCircleSection, ThinCircleSection

TOLERANCE = 1e-9
PI = np.pi


@pytest.mark.order(2)
@pytest.mark.dependency(
    depends=[
        "tests/test_material.py::test_end",
        "tests/test_profile.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


@pytest.mark.dependency(depends=["test_begin"])
class TestSection(object):
    pass


@pytest.mark.dependency(depends=["test_begin"])
class TestCircleSection(TestSection):
    def setup_random(self):
        E = np.random.uniform(100, 200)
        nu = np.random.uniform(0.01, 0.49)
        material = Isotropic(E=E, nu=nu)
        R = np.random.uniform(1, 2)
        profile = Circle(R=R)
        self.section = CircleSection(material, profile)

    def compute_correct_areas(self):
        nu = self.section.material.nu
        R = self.section.profile.R
        coef = 6 * (1 + nu) / (7 + 6 * nu)
        Ax = PI * R**2
        Ay = coef * Ax
        Az = coef * Ax
        self.goodA = (Ax, Ay, Az)

    def compute_correct_inertias(self):
        R = self.section.profile.R
        Iy = PI * R**4 / 4
        Iz = Iy
        Ix = 2 * Iy
        self.goodI = (Ix, Iy, Iz)

    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_random_areas(self, ntests=10):
        for i in range(ntests):
            self.setup_random()
            self.compute_correct_areas()
            np.testing.assert_almost_equal(self.section.A, self.goodA)

    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_random_inertias(self, ntests=10):
        for i in range(ntests):
            self.setup_random()
            self.compute_correct_inertias()
            np.testing.assert_almost_equal(self.section.I, self.goodI)

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=[
            "TestCircleSection::test_random_areas",
            "TestCircleSection::test_random_inertias",
        ]
    )
    def test_end(self):
        pass


@pytest.mark.dependency(depends=["test_begin"])
class TestHollowCircleSection(TestSection):
    def setup_random(self):
        E = np.random.uniform(100, 200)
        nu = np.random.uniform(0.01, 0.49)
        material = Isotropic(E=E, nu=nu)
        R = np.random.uniform(1, 2)
        e = np.random.uniform(0.1, 0.3) * R
        Ri = R - 0.5 * e
        Re = R + 0.5 * e
        profile = HollowCircle(Ri=Ri, Re=Re)
        self.section = HollowCircleSection(material, profile)

    def compute_correct_areas(self):
        nu = self.section.material.nu
        Ri = self.section.profile.Ri
        Re = self.section.profile.Re
        m2 = (Ri / Re) ** 2
        numerator = 6 * (1 + nu) * (1 + m2) ** 2
        denominator = (7 + 6 * nu) * (1 + m2) ** 2 + 4 * m2 * (5 + 3 * nu)
        coef = numerator / denominator
        Ax = PI * (Re**2 - Ri**2)
        Ay = coef * Ax
        Az = coef * Ax
        self.goodA = (Ax, Ay, Az)

    def compute_correct_inertias(self):
        Ri = self.section.profile.Ri
        Re = self.section.profile.Re
        Iy = PI * (Re**4 - Ri**4) / 4
        Iz = Iy
        Ix = 2 * Iy
        self.goodI = (Ix, Iy, Iz)

    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_random_areas(self, ntests=10):
        for i in range(ntests):
            self.setup_random()
            self.compute_correct_areas()
            np.testing.assert_almost_equal(self.section.A, self.goodA)

    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_random_inertias(self, ntests=10):
        for i in range(ntests):
            self.setup_random()
            self.compute_correct_inertias()
            np.testing.assert_almost_equal(self.section.I, self.goodI)

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=[
            "TestHollowCircleSection::test_random_areas",
            "TestHollowCircleSection::test_random_inertias",
        ]
    )
    def test_end(self):
        pass


@pytest.mark.dependency(depends=["test_begin"])
class TestThinCircleSection(TestSection):
    def setup_random(self):
        E = np.random.uniform(100, 200)
        nu = np.random.uniform(0.01, 0.49)
        material = Isotropic(E=E, nu=nu)
        R = np.random.uniform(1, 2)
        e = 0.02 * R
        Ri = R - 0.5 * e
        Re = R + 0.5 * e
        profile = HollowCircle(Ri=Ri, Re=Re)
        self.section = ThinCircleSection(material, profile)

    def compute_correct_areas(self):
        nu = self.section.material.nu
        R = self.section.profile.R
        e = self.section.profile.e
        coef = 2 * (1 + nu) / (4 + 3 * nu)
        Ax = 2 * PI * e * R
        Ay = coef * Ax
        Az = coef * Ax
        self.goodA = (Ax, Ay, Az)

    def compute_correct_inertias(self):
        R = self.section.profile.R
        e = self.section.profile.e
        Iy = PI * e * R**3
        Iz = Iy
        Ix = 2 * Iy
        self.goodI = (Ix, Iy, Iz)

    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_random_areas(self, ntests=10):
        for i in range(ntests):
            self.setup_random()
            self.compute_correct_areas()
            np.testing.assert_almost_equal(self.section.A, self.goodA)

    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_random_inertias(self, ntests=10):
        for i in range(ntests):
            self.setup_random()
            self.compute_correct_inertias()
            np.testing.assert_almost_equal(self.section.I, self.goodI)

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=[
            "TestThinCircleSection::test_random_areas",
            "TestThinCircleSection::test_random_inertias",
        ]
    )
    def test_end(self):
        pass


class TestRetangularSection:
    @pytest.mark.order(2)
    @pytest.mark.dependency()
    def test_end(self):
        pass


class TestHollowRetangularSection:
    @pytest.mark.order(2)
    @pytest.mark.dependency()
    def test_end(self):
        pass


class TestThinRetangularSection:
    @pytest.mark.order(2)
    @pytest.mark.dependency()
    def test_end(self):
        pass


class TestSquareSection:
    @pytest.mark.order(2)
    @pytest.mark.dependency()
    def test_end(self):
        pass


class TestHollowSquareSection:
    @pytest.mark.order(2)
    @pytest.mark.dependency()
    def test_end(self):
        pass


class TestThinSquareSection:
    @pytest.mark.order(2)
    @pytest.mark.dependency()
    def test_end(self):
        pass


@pytest.mark.order(2)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "TestCircleSection::test_end",
        "TestHollowCircleSection::test_end",
        "TestThinCircleSection::test_end",
        "TestRetangularSection::test_end",
        "TestHollowRetangularSection::test_end",
        "TestThinRetangularSection::test_end",
        "TestSquareSection::test_end",
        "TestHollowSquareSection::test_end",
        "TestThinSquareSection::test_end",
    ]
)
def test_end():
    pass
