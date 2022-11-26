from typing import Optional, Tuple

import numpy as np
import pytest

from compmec.strct.element import EulerBernoulli
from compmec.strct.material import Isotropic
from compmec.strct.profile import Circle
from compmec.strct.section import CircleSection
from compmec.strct.solver import solve


@pytest.mark.order(6)
@pytest.mark.dependency(
    depends=[
        "tests/test_solver.py::test_end",
        "tests/test_material.py::test_end",
        "tests/test_structural1D.py::test_end",
        "tests/test_section.py::TestCircleSection::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


class InitRodBeamEuler(object):
    def get_random_unit_vector(
        self, directionstr: str, perpendicular_to: Optional[Tuple[float]] = None
    ):
        if not isinstance(directionstr, str):
            raise TypeError
        if perpendicular_to is not None:
            if not isinstance(perpendicular_to, str):
                raise TypeError
        vector = np.random.uniform(-1, 1, 3)
        for i, s in enumerate(["x", "y", "z"]):
            if s not in directionstr:
                vector[i] = 0
        norm = np.linalg.norm(vector)
        if perpendicular_to is None:
            return vector / norm
        perpendicular_to /= np.linalg.norm(perpendicular_to)
        vector -= np.inner(vector, perpendicular_to) * perpendicular_to
        norm = np.linalg.norm(vector)
        return vector / norm

    def set_random_material(self):
        E = np.random.uniform(1, 2)
        nu = np.random.uniform(0, 0.49)
        self.material = Isotropic(E=E, nu=nu)

    def set_random_profile(self):
        R = np.random.uniform(1, 2)
        self.profile = Circle(R=R)

    def set_random_section(self):
        self.set_random_material()
        self.set_random_profile()
        self.section = CircleSection(self.material, self.profile)

    def set_random_beam(self):
        direction = np.array(self.direction_beam) / np.linalg.norm(self.direction_beam)
        self.lenght = np.random.uniform(1, 2)
        A = (0, 0, 0)
        B = tuple(self.lenght * direction)
        self.beam = EulerBernoulli([A, B])

        self.set_random_section()
        self.beam.material = self.material
        self.beam.section = self.section


class TestOneRodBeamTraction(InitRodBeamEuler):
    def compute_analitic_displacement_field(self):
        A = self.profile.A
        P = self.charge
        L = self.lenght
        E = self.material.E
        return P * L / (A * E)

    def run_test(self):
        self.set_random_beam()
        self.charge = np.random.uniform(-1, 1)

        P = self.charge
        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))
        U[0, :] = 0
        F[1, :3] = P * self.direction_beam

        K = self.beam.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)

        ur = self.compute_analitic_displacement_field()
        Ugood = np.zeros((2, 6))
        Fgood = np.zeros((2, 6))
        Ugood[1, :3] = ur * self.direction_beam
        Fgood[0, :3] = -P * self.direction_beam
        Fgood[1, :3] = P * self.direction_beam

        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)

    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_random_x(self, ntests=10):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("x")
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_random_y(self, ntests=10):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("y")
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_random_z(self, ntests=10):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("z")
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamTraction::test_random_x",
            "TestOneRodBeamTraction::test_random_y",
        ]
    )
    def test_random_xy(self, ntests=10):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xy")
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamTraction::test_random_y",
            "TestOneRodBeamTraction::test_random_z",
        ]
    )
    def test_random_yz(self, ntests=10):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("yz")
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamTraction::test_random_x",
            "TestOneRodBeamTraction::test_random_z",
        ]
    )
    def test_random_xz(self, ntests=10):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xz")
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamTraction::test_random_xy",
            "TestOneRodBeamTraction::test_random_yz",
            "TestOneRodBeamTraction::test_random_xz",
        ]
    )
    def test_random_xyz(self, ntests=10):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xyz")
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["TestOneRodBeamTraction::test_random_xyz"])
    def test_end(self):
        pass


class TestOneRodBeamTorsion(InitRodBeamEuler):
    def compute_analitic_rotation_field(self):
        Ix = self.section.I[0]
        T = self.charge
        L = self.lenght
        G = self.material.G
        return T * L / (G * Ix)

    def run_test(self):
        self.set_random_beam()
        self.charge = np.random.uniform(-1, 1)

        T = self.charge
        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))
        U[0, :] = 0
        F[1, 3:] = T * self.direction_beam

        K = self.beam.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)

        tr = self.compute_analitic_rotation_field()
        Ugood = np.zeros((2, 6))
        Fgood = np.zeros((2, 6))
        Ugood[1, 3:] = tr * self.direction_beam
        Fgood[0, 3:] = -T * self.direction_beam
        Fgood[1, 3:] = T * self.direction_beam

        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)

    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_random_x(self, ntests=10):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("x")
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_random_y(self, ntests=10):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("y")
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_random_z(self, ntests=10):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("z")
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamTorsion::test_random_x",
            "TestOneRodBeamTorsion::test_random_y",
        ]
    )
    def test_random_xy(self, ntests=10):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xy")
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamTorsion::test_random_y",
            "TestOneRodBeamTorsion::test_random_z",
        ]
    )
    def test_random_yz(self, ntests=10):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("yz")
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamTorsion::test_random_x",
            "TestOneRodBeamTorsion::test_random_z",
        ]
    )
    def test_random_xz(self, ntests=10):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xz")
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamTorsion::test_random_xy",
            "TestOneRodBeamTorsion::test_random_yz",
            "TestOneRodBeamTorsion::test_random_xz",
        ]
    )
    def test_random_xyz(self, ntests=10):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xyz")
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["TestOneRodBeamTorsion::test_random_xyz"])
    def test_end(self):
        pass


@pytest.mark.order(6)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "TestOneRodBeamTraction::test_end",
        "TestOneRodBeamTorsion::test_end",
    ]
)
def test_end():
    pass
