from typing import Optional, Tuple

import numpy as np
import pytest

from compmec.strct.element import Truss
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


class InitRodTruss(object):
    def get_random_unit_vector(self, directionstr: str) -> Tuple[float]:
        vector = np.random.uniform(-1, 1, 3)
        vector = vector.astype("float64")
        for i, s in enumerate(["x", "y", "z"]):
            if s not in directionstr:
                vector[i] = 0
        norm = np.linalg.norm(vector)
        return vector / norm

    def set_random_material(self):
        E = np.random.uniform(1, 2)
        nu = np.random.uniform(0, 0.49)
        self.material = Isotropic(E=E, nu=nu)

    def set_random_profile(self):
        R = np.random.uniform(1, 2)
        self.profile = Circle(radius=R)

    def set_random_section(self):
        self.set_random_material()
        self.set_random_profile()
        self.section = CircleSection(self.material, self.profile)

    def set_random_beam(self):
        direction = np.array(self.direction_beam) / np.linalg.norm(self.direction_beam)
        self.lenght = np.random.uniform(1, 2)
        A = (0, 0, 0)
        B = tuple(self.lenght * direction)
        self.truss = Truss([A, B])

        self.set_random_section()
        self.truss.section = self.section

    def run_test(self):
        self.set_random_beam()
        self.charge = np.random.uniform(-1, 1)

        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))
        U[0, :] = 0
        F[1, :3] = self.apply_force()
        F[1, 3:] = self.apply_momentum()

        K = self.truss.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)
        self.compare_solution(Utest, Ftest)

    def compare_solution(self, Utest, Ftest):
        Ugood = np.zeros((2, 6))
        Fgood = np.zeros((2, 6))
        Ugood[1, :3] = self.compute_analitic_displacement_field()
        Ugood[1, 3:] = self.compute_analitic_rotation_field()
        Fgood[0, :3] = self.compute_analitic_reaction_force()
        Fgood[0, 3:] = self.compute_analitic_reaction_momentum()
        Fgood[1, :3] = self.apply_force()
        Fgood[1, 3:] = self.apply_momentum()

        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)


class TestOneRodTrussTraction(InitRodTruss):
    def compute_analitic_displacement_field(self):
        A = self.profile.area
        P = self.charge
        L = self.lenght
        E = self.material.E
        return P * L * self.direction_beam / (A * E)

    def compute_analitic_rotation_field(self):
        return (0, 0, 0)

    def compute_analitic_reaction_force(self):
        return -self.charge * self.direction_beam

    def compute_analitic_reaction_momentum(self):
        return (0, 0, 0)

    def apply_force(self):
        return self.charge * self.direction_beam

    def apply_momentum(self):
        return (0, 0, 0)

    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_random_x(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("x")
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_random_y(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("y")
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_random_z(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("z")
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodTrussTraction::test_random_x",
            "TestOneRodTrussTraction::test_random_y",
        ]
    )
    def test_random_xy(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xy")
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodTrussTraction::test_random_y",
            "TestOneRodTrussTraction::test_random_z",
        ]
    )
    def test_random_yz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("yz")
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodTrussTraction::test_random_x",
            "TestOneRodTrussTraction::test_random_z",
        ]
    )
    def test_random_xz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xz")
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodTrussTraction::test_random_xy",
            "TestOneRodTrussTraction::test_random_yz",
            "TestOneRodTrussTraction::test_random_xz",
        ]
    )
    def test_random_xyz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xyz")
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["TestOneRodTrussTraction::test_random_xyz"])
    def test_end(self):
        pass


@pytest.mark.order(6)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "TestOneRodTrussTraction::test_end",
    ]
)
def test_end():
    pass
