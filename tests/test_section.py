from typing import List, TypeVar

import numpy as np
import pytest

from compmec.strct import profile as prof
from compmec.strct import section as sect
from compmec.strct.material import Isotropic

TOLERANCE = 1e-9
PI = np.pi


@pytest.mark.order(2)
@pytest.mark.dependency(
    depends=[
        "tests/test_material.py::test_end",
        "tests/test_geometry.py::test_end",
        "tests/test_profile.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


class TestSection:
    pass


class TestCircleSection(TestSection):
    def setup_random(self):
        E = np.random.uniform(100, 200)
        nu = np.random.uniform(0.01, 0.49)
        material = Isotropic(E=E, nu=nu)
        R = np.random.uniform(1, 2)
        profile = prof.Circle(radius=R)
        self.section = sect.CircleSection(material, profile)

    def compute_correct_areas(self):
        nu = self.section.material.nu
        R = self.section.profile.radius
        coef = 6 * (1 + nu) / (7 + 6 * nu)
        Ax = PI * R**2
        Ay = coef * Ax
        Az = coef * Ax
        self.goodA = (Ax, Ay, Az)

    def compute_correct_inertias(self):
        R = self.section.profile.radius
        Iy = PI * R**4 / 4
        Iz = Iy
        Ix = 2 * Iy
        self.goodI = (Ix, Iy, Iz)

    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["TestCircleSection::test_begin"])
    def test_creation(self, ntests=10):
        for i in range(ntests):
            self.setup_random()
            self.compute_correct_areas()
            np.testing.assert_almost_equal(self.section.A, self.goodA)

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=["TestCircleSection::test_begin", "TestCircleSection::test_creation"]
    )
    def test_random_areas(self, ntests=10):
        for i in range(ntests):
            self.setup_random()
            self.compute_correct_areas()
            np.testing.assert_almost_equal(self.section.A, self.goodA)

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=["TestCircleSection::test_begin", "TestCircleSection::test_creation"]
    )
    def test_random_inertias(self, ntests=10):
        for i in range(ntests):
            self.setup_random()
            self.compute_correct_inertias()
            np.testing.assert_almost_equal(self.section.I, self.goodI)

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=[
            "TestCircleSection::test_begin",
            "TestCircleSection::test_creation",
            "TestCircleSection::test_random_areas",
            "TestCircleSection::test_random_inertias",
        ]
    )
    def test_end(self):
        pass


class TestHollowCircleSection(TestSection):
    def setup_random(self):
        E = np.random.uniform(100, 200)
        nu = np.random.uniform(0.01, 0.49)
        material = Isotropic(E=E, nu=nu)
        R = np.random.uniform(1, 2)
        e = np.random.uniform(0.1, 0.3) * R
        Ri = R - 0.5 * e
        Re = R + 0.5 * e
        profile = prof.HollowCircle(Ri=Ri, Re=Re)
        self.section = sect.HollowCircleSection(material, profile)

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
    def test_begin(self):
        pass

    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["TestHollowCircleSection::test_begin"])
    def test_creation(self):
        pass

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=[
            "TestHollowCircleSection::test_begin",
            "TestHollowCircleSection::test_creation",
        ]
    )
    def test_random_areas(self, ntests=10):
        for i in range(ntests):
            self.setup_random()
            self.compute_correct_areas()
            np.testing.assert_almost_equal(self.section.A, self.goodA)

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=[
            "TestHollowCircleSection::test_begin",
            "TestHollowCircleSection::test_creation",
        ]
    )
    def test_random_inertias(self, ntests=10):
        for i in range(ntests):
            self.setup_random()
            self.compute_correct_inertias()
            np.testing.assert_almost_equal(self.section.I, self.goodI)

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=[
            "TestHollowCircleSection::test_begin",
            "TestHollowCircleSection::test_creation",
            "TestHollowCircleSection::test_random_areas",
            "TestHollowCircleSection::test_random_inertias",
        ]
    )
    def test_end(self):
        pass


class TestRetangularSection:
    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["TestRetangularSection::test_begin"])
    def test_creation(self):
        retangular = prof.Retangular(2, 3)
        steel = Isotropic(E=210e3, nu=0.3)
        sect.RetangularSection(steel, retangular)

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=[
            "TestRetangularSection::test_begin",
            "TestRetangularSection::test_creation",
        ]
    )
    def test_end(self):
        pass


class TestHollowRetangularSection:
    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["TestHollowRetangularSection::test_begin"])
    def test_creation(self):
        retangular = prof.HollowRetangular(4, 4, 5, 5)
        steel = Isotropic(E=210e3, nu=0.3)
        sect.HollowRetangularSection(steel, retangular)

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=[
            "TestHollowRetangularSection::test_begin",
            "TestHollowRetangularSection::test_creation",
        ]
    )
    def test_end(self):
        pass


class TestPerfilISection:
    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["TestPerfilISection::test_begin"])
    def test_creation(self):
        profile = prof.PerfilI(3, 4, 1, 1)
        steel = Isotropic(E=210e3, nu=0.3)
        sect.PerfilISection(steel, profile)

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=["TestPerfilISection::test_begin", "TestPerfilISection::test_creation"]
    )
    def test_areas(self):
        profile = prof.PerfilI(4, 8, 3, 2)
        steel = Isotropic(E=210e3, nu=0.3)
        section = sect.PerfilISection(steel, profile)
        coeff = 43940 / 80343
        assert section.A[0] == 34
        assert section.A[1] == 34 * coeff
        assert section.A[2] == 34 * coeff

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=["TestPerfilISection::test_begin", "TestPerfilISection::test_creation"]
    )
    def test_inertias(self):
        profile = prof.PerfilI(4, 8, 3, 2)
        steel = Isotropic(E=210e3, nu=0.3)
        section = sect.PerfilISection(steel, profile)
        assert section.I[1] == 106 / 3
        assert section.I[2] == 2537 / 6

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=[
            "TestPerfilISection::test_begin",
            "TestPerfilISection::test_creation",
            "TestPerfilISection::test_areas",
            "TestPerfilISection::test_inertias",
        ]
    )
    def test_end(self):
        pass


class TestGeneralSection:
    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["TestGeneralSection::test_begin"])
    def test_creation(self):
        section = sect.GeneralSection()

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=["TestGeneralSection::test_begin", "TestGeneralSection::test_creation"]
    )
    def test_set_get(self):
        section = sect.GeneralSection()
        with pytest.raises(ValueError):
            section.A
        with pytest.raises(ValueError):
            section.I
        with pytest.raises(ValueError):
            section.A = (-1, 1, 1)
        with pytest.raises(ValueError):
            section.A = (0, 1, 1)
        with pytest.raises(ValueError):
            section.A = (0, 1, 1, 4)
        with pytest.raises(ValueError):
            section.A = [(0, 1, 1, 4)]
        with pytest.raises(ValueError):
            section.I = (-1, 1, 1)
        with pytest.raises(ValueError):
            section.I = (0, 1, 1)
        with pytest.raises(ValueError):
            section.I = (0, 1, 1, 4)
        with pytest.raises(ValueError):
            section.I = [(0, 1, 1, 4)]
        section.A = (1, 3, 4)
        section.I = (5, 8, 9)
        np.testing.assert_allclose(section.A, (1, 3, 4))
        np.testing.assert_allclose(section.I, (5, 8, 9))

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=[
            "TestGeneralSection::test_begin",
            "TestGeneralSection::test_creation",
            "TestGeneralSection::test_set_get",
            "TestGeneralSection::test_begin",
        ]
    )
    def test_end(self):
        pass


class TestFail:
    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=[
            "test_begin",
            "TestCircleSection::test_end",
            "TestHollowCircleSection::test_end",
            "TestRetangularSection::test_end",
            "TestHollowRetangularSection::test_end",
            "TestPerfilISection::test_end",
            "TestGeneralSection::test_end",
        ]
    )
    def test_begin(self):
        pass

    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["TestFail::test_begin"])
    def test_fail1(self):
        steel = Isotropic(E=210e3, nu=0.3)
        circle = prof.Circle(diameter=3)
        with pytest.raises(TypeError):
            sect.CircleSection(1, circle)
        with pytest.raises(TypeError):
            sect.CircleSection(steel, 1)
        with pytest.raises(TypeError):
            sect.CircleSection("asd", circle)
        with pytest.raises(TypeError):
            sect.CircleSection(steel, "asd")
        holcirc = prof.HollowCircle(1, 2)
        with pytest.raises(TypeError):
            sect.HollowCircleSection(1, circle)
        with pytest.raises(TypeError):
            sect.HollowCircleSection(steel, 1)
        with pytest.raises(TypeError):
            sect.HollowCircleSection("asd", circle)
        with pytest.raises(TypeError):
            sect.HollowCircleSection(steel, "asd")

    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["TestFail::test_begin", "TestFail::test_fail1"])
    def test_end(self):
        pass


@pytest.mark.order(2)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "TestCircleSection::test_end",
        "TestHollowCircleSection::test_end",
        "TestRetangularSection::test_end",
        "TestHollowRetangularSection::test_end",
        "TestPerfilISection::test_end",
        "TestGeneralSection::test_end",
        "TestFail::test_end",
    ]
)
def test_end():
    pass
