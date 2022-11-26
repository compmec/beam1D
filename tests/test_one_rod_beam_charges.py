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
        vector = np.random.uniform(-1, 1, 3)
        vector = vector.astype("float64")
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

    def run_test(self):
        self.set_random_beam()
        self.charge = np.random.uniform(-1, 1)

        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))
        U[0, :] = 0
        F[1, :3] = self.apply_force()
        F[1, 3:] = self.apply_momentum()

        K = self.beam.stiffness_matrix()
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


class TestOneRodBeamTraction(InitRodBeamEuler):
    def compute_analitic_displacement_field(self):
        A = self.profile.A
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
            "TestOneRodBeamTraction::test_random_x",
            "TestOneRodBeamTraction::test_random_y",
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
            "TestOneRodBeamTraction::test_random_y",
            "TestOneRodBeamTraction::test_random_z",
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
            "TestOneRodBeamTraction::test_random_x",
            "TestOneRodBeamTraction::test_random_z",
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
            "TestOneRodBeamTraction::test_random_xy",
            "TestOneRodBeamTraction::test_random_yz",
            "TestOneRodBeamTraction::test_random_xz",
        ]
    )
    def test_random_xyz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xyz")
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["TestOneRodBeamTraction::test_random_xyz"])
    def test_end(self):
        pass


class TestOneRodBeamTorsion(InitRodBeamEuler):
    def compute_analitic_displacement_field(self):
        return (0, 0, 0)

    def compute_analitic_rotation_field(self):
        Ix = self.section.I[0]
        T = self.charge
        L = self.lenght
        G = self.material.G
        return T * L * self.direction_beam / (G * Ix)

    def compute_analitic_reaction_force(self):
        return (0, 0, 0)

    def compute_analitic_reaction_momentum(self):
        return -self.charge * self.direction_beam

    def apply_force(self):
        return (0, 0, 0)

    def apply_momentum(self):
        return self.charge * self.direction_beam

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
            "TestOneRodBeamTorsion::test_random_x",
            "TestOneRodBeamTorsion::test_random_y",
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
            "TestOneRodBeamTorsion::test_random_y",
            "TestOneRodBeamTorsion::test_random_z",
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
            "TestOneRodBeamTorsion::test_random_x",
            "TestOneRodBeamTorsion::test_random_z",
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
            "TestOneRodBeamTorsion::test_random_xy",
            "TestOneRodBeamTorsion::test_random_yz",
            "TestOneRodBeamTorsion::test_random_xz",
        ]
    )
    def test_random_xyz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xyz")
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["TestOneRodBeamTorsion::test_random_xyz"])
    def test_end(self):
        pass


class TestOneRodBeamBendingForce(InitRodBeamEuler):
    def compute_analitic_displacement_field(self):
        Iy = self.section.I[1]
        P = self.charge
        L = self.lenght
        E = self.material.E
        return P * L**3 * self.direction_force / (3 * E * Iy)

    def compute_analitic_rotation_field(self):
        Iy = self.section.I[1]
        P = self.charge
        L = self.lenght
        E = self.material.E
        direction_momentum = np.cross(self.direction_beam, self.direction_force)
        direction_momentum /= np.linalg.norm(direction_momentum)
        return P * L**2 * direction_momentum / (2 * E * Iy)

    def compute_analitic_reaction_force(self):
        return -self.charge * self.direction_force

    def compute_analitic_reaction_momentum(self):
        direction_momentum = np.cross(self.direction_beam, self.direction_force)
        direction_momentum /= np.linalg.norm(direction_momentum)
        return -self.charge * self.lenght * direction_momentum

    def apply_force(self):
        return self.charge * self.direction_force

    def apply_momentum(self):
        return (0, 0, 0)

    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_random_x2y(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("x")
            self.direction_force = self.get_random_unit_vector("y", self.direction_beam)
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=["test_begin", "TestOneRodBeamBendingForce::test_random_x2y"]
    )
    def test_random_x2z(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("x")
            self.direction_force = self.get_random_unit_vector("z", self.direction_beam)
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=["test_begin", "TestOneRodBeamBendingForce::test_random_x2y"]
    )
    def test_random_y2x(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("y")
            self.direction_force = self.get_random_unit_vector("x", self.direction_beam)
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=["test_begin", "TestOneRodBeamBendingForce::test_random_x2y"]
    )
    def test_random_y2z(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("y")
            self.direction_force = self.get_random_unit_vector("z", self.direction_beam)
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=["test_begin", "TestOneRodBeamBendingForce::test_random_x2y"]
    )
    def test_random_z2x(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("z")
            self.direction_force = self.get_random_unit_vector("x", self.direction_beam)
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=["test_begin", "TestOneRodBeamBendingForce::test_random_x2y"]
    )
    def test_random_z2y(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("z")
            self.direction_force = self.get_random_unit_vector("y", self.direction_beam)
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingForce::test_random_x2y",
            "TestOneRodBeamBendingForce::test_random_x2z",
        ]
    )
    def test_random_x2yz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("x")
            self.direction_force = self.get_random_unit_vector(
                "yz", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingForce::test_random_y2x",
            "TestOneRodBeamBendingForce::test_random_y2z",
        ]
    )
    def test_random_y2xz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("y")
            self.direction_force = self.get_random_unit_vector(
                "xz", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingForce::test_random_z2x",
            "TestOneRodBeamBendingForce::test_random_z2y",
        ]
    )
    def test_random_z2xy(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("z")
            self.direction_force = self.get_random_unit_vector(
                "xy", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingForce::test_random_x2y",
            "TestOneRodBeamBendingForce::test_random_y2x",
        ]
    )
    def test_random_xy2xy(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xy")
            self.direction_force = self.get_random_unit_vector(
                "xy", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingForce::test_random_x2z",
            "TestOneRodBeamBendingForce::test_random_z2x",
        ]
    )
    def test_random_xz2xz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xz")
            self.direction_force = self.get_random_unit_vector(
                "xz", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingForce::test_random_y2z",
            "TestOneRodBeamBendingForce::test_random_z2y",
        ]
    )
    def test_random_yz2yz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("yz")
            self.direction_force = self.get_random_unit_vector(
                "yz", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingForce::test_random_xy2xy",
            "TestOneRodBeamBendingForce::test_random_x2yz",
            "TestOneRodBeamBendingForce::test_random_y2xz",
        ]
    )
    def test_random_xy2xyz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xy")
            self.direction_force = self.get_random_unit_vector(
                "xyz", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingForce::test_random_yz2yz",
            "TestOneRodBeamBendingForce::test_random_y2xz",
            "TestOneRodBeamBendingForce::test_random_z2xy",
        ]
    )
    def test_random_yz2xyz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("yz")
            self.direction_force = self.get_random_unit_vector(
                "xyz", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingForce::test_random_xz2xz",
            "TestOneRodBeamBendingForce::test_random_x2yz",
            "TestOneRodBeamBendingForce::test_random_z2xy",
        ]
    )
    def test_random_xz2xyz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xz")
            self.direction_force = self.get_random_unit_vector(
                "xyz", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingForce::test_random_xy2xyz",
            "TestOneRodBeamBendingForce::test_random_xz2xyz",
            "TestOneRodBeamBendingForce::test_random_xz2xyz",
        ]
    )
    def test_random_xyz2xyz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xyz")
            self.direction_force = self.get_random_unit_vector(
                "xyz", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["TestOneRodBeamBendingForce::test_random_xyz2xyz"])
    def test_end(self):
        pass


class TestOneRodBeamBendingMomentum(InitRodBeamEuler):
    def compute_analitic_displacement_field(self):
        Iy = self.section.I[1]
        M = self.charge
        L = self.lenght
        E = self.material.E

        direction_displacement = np.cross(self.direction_momentum, self.direction_beam)
        direction_displacement /= np.linalg.norm(direction_displacement)
        return M * L**2 * direction_displacement / (2 * E * Iy)

    def compute_analitic_rotation_field(self):
        Iy = self.section.I[1]
        M = self.charge
        L = self.lenght
        E = self.material.E
        return M * L * self.direction_momentum / (E * Iy)

    def compute_analitic_reaction_force(self):
        return (0, 0, 0)

    def compute_analitic_reaction_momentum(self):
        return -self.charge * self.direction_momentum

    def apply_force(self):
        return (0, 0, 0)

    def apply_momentum(self):
        return self.charge * self.direction_momentum

    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_random_x2y(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("x")
            self.direction_momentum = self.get_random_unit_vector(
                "y", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=["test_begin", "TestOneRodBeamBendingMomentum::test_random_x2y"]
    )
    def test_random_x2z(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("x")
            self.direction_momentum = self.get_random_unit_vector(
                "z", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=["test_begin", "TestOneRodBeamBendingMomentum::test_random_x2y"]
    )
    def test_random_y2x(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("y")
            self.direction_momentum = self.get_random_unit_vector(
                "x", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=["test_begin", "TestOneRodBeamBendingMomentum::test_random_x2y"]
    )
    def test_random_y2z(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("y")
            self.direction_momentum = self.get_random_unit_vector(
                "z", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=["test_begin", "TestOneRodBeamBendingMomentum::test_random_x2y"]
    )
    def test_random_z2x(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("z")
            self.direction_momentum = self.get_random_unit_vector(
                "x", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=["test_begin", "TestOneRodBeamBendingMomentum::test_random_x2y"]
    )
    def test_random_z2y(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("z")
            self.direction_momentum = self.get_random_unit_vector(
                "y", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingMomentum::test_random_x2y",
            "TestOneRodBeamBendingMomentum::test_random_x2z",
        ]
    )
    def test_random_x2yz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("x")
            self.direction_momentum = self.get_random_unit_vector(
                "yz", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingMomentum::test_random_y2x",
            "TestOneRodBeamBendingMomentum::test_random_y2z",
        ]
    )
    def test_random_y2xz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("y")
            self.direction_momentum = self.get_random_unit_vector(
                "xz", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingMomentum::test_random_z2x",
            "TestOneRodBeamBendingMomentum::test_random_z2y",
        ]
    )
    def test_random_z2xy(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("z")
            self.direction_momentum = self.get_random_unit_vector(
                "xy", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingMomentum::test_random_x2y",
            "TestOneRodBeamBendingMomentum::test_random_y2x",
        ]
    )
    def test_random_xy2xy(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xy")
            self.direction_momentum = self.get_random_unit_vector(
                "xy", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingMomentum::test_random_x2z",
            "TestOneRodBeamBendingMomentum::test_random_z2x",
        ]
    )
    def test_random_xz2xz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xz")
            self.direction_momentum = self.get_random_unit_vector(
                "xz", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingMomentum::test_random_y2z",
            "TestOneRodBeamBendingMomentum::test_random_z2y",
        ]
    )
    def test_random_yz2yz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("yz")
            self.direction_momentum = self.get_random_unit_vector(
                "yz", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingMomentum::test_random_xy2xy",
            "TestOneRodBeamBendingMomentum::test_random_x2yz",
            "TestOneRodBeamBendingMomentum::test_random_y2xz",
        ]
    )
    def test_random_xy2xyz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xy")
            self.direction_momentum = self.get_random_unit_vector(
                "xyz", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingMomentum::test_random_yz2yz",
            "TestOneRodBeamBendingMomentum::test_random_y2xz",
            "TestOneRodBeamBendingMomentum::test_random_z2xy",
        ]
    )
    def test_random_yz2xyz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("yz")
            self.direction_momentum = self.get_random_unit_vector(
                "xyz", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingMomentum::test_random_xz2xz",
            "TestOneRodBeamBendingMomentum::test_random_x2yz",
            "TestOneRodBeamBendingMomentum::test_random_z2xy",
        ]
    )
    def test_random_xz2xyz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xz")
            self.direction_momentum = self.get_random_unit_vector(
                "xyz", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamBendingMomentum::test_random_xy2xyz",
            "TestOneRodBeamBendingMomentum::test_random_xz2xyz",
            "TestOneRodBeamBendingMomentum::test_random_xz2xyz",
        ]
    )
    def test_random_xyz2xyz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xyz")
            self.direction_momentum = self.get_random_unit_vector(
                "xyz", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=["TestOneRodBeamBendingMomentum::test_random_xyz2xyz"]
    )
    def test_end(self):
        pass


class TestOneRodBeamAllCharges(InitRodBeamEuler):
    def compute_analitic_displacement_field(self):
        force_charge = self.force_value * self.direction_force
        momentum_charge = self.momentum_value * self.direction_momentum
        traction_force_value = np.inner(force_charge, self.direction_beam)
        direction_bending_force = self.direction_force - self.direction_beam * np.inner(
            self.direction_force, self.direction_beam
        )
        direction_bending_force /= np.linalg.norm(direction_bending_force)
        direction_bending_momentum = (
            self.direction_momentum
            - self.direction_beam
            * np.inner(self.direction_momentum, self.direction_beam)
        )
        direction_bending_momentum /= np.linalg.norm(direction_bending_momentum)
        bending_force_value = np.inner(force_charge, direction_bending_force)
        bending_momentum_value = np.inner(momentum_charge, direction_bending_momentum)
        direction_disp_bend_momentum = np.cross(
            direction_bending_momentum, self.direction_beam
        )

        A = self.section.A[0]
        Iy = self.section.I[1]
        L = self.lenght
        E = self.material.E

        displacement = traction_force_value * L / (A * E) * self.direction_beam
        displacement += (
            bending_force_value * L**3 / (3 * E * Iy) * direction_bending_force
        )
        displacement += (
            bending_momentum_value
            * L**2
            / (2 * E * Iy)
            * direction_disp_bend_momentum
        )
        return displacement

    def compute_analitic_rotation_field(self):
        force_charge = self.force_value * self.direction_force
        momentum_charge = self.momentum_value * self.direction_momentum
        torsion_momentum_value = np.inner(momentum_charge, self.direction_beam)
        direction_bending_force = self.direction_force - self.direction_beam * np.inner(
            self.direction_force, self.direction_beam
        )
        direction_bending_force /= np.linalg.norm(direction_bending_force)
        direction_bending_momentum = (
            self.direction_momentum
            - self.direction_beam
            * np.inner(self.direction_momentum, self.direction_beam)
        )
        direction_bending_momentum /= np.linalg.norm(direction_bending_momentum)
        bending_force_value = np.inner(force_charge, direction_bending_force)
        bending_momentum_value = np.inner(momentum_charge, direction_bending_momentum)
        direction_rot_bend_force = np.cross(
            self.direction_beam, direction_bending_force
        )

        A = self.section.A[0]
        Iy = self.section.I[1]
        J = self.section.I[0]
        L = self.lenght
        E = self.material.E
        G = self.material.G

        rotation = torsion_momentum_value * L / (J * G) * self.direction_beam
        rotation += (
            bending_force_value * L**2 / (2 * E * Iy) * direction_rot_bend_force
        )
        rotation += bending_momentum_value * L / (E * Iy) * direction_bending_momentum
        return rotation

    def compute_analitic_reaction_force(self):
        return -self.force_value * self.direction_force

    def compute_analitic_reaction_momentum(self):
        L = self.lenght
        force_charge = self.force_value * self.direction_force
        momentum_charge = self.momentum_value * self.direction_momentum
        reaction_momentum = -momentum_charge
        reaction_momentum -= np.cross(L * self.direction_beam, force_charge)
        return reaction_momentum

    def apply_force(self):
        return self.force_value * self.direction_force

    def apply_momentum(self):
        return self.momentum_value * self.direction_momentum

    @pytest.mark.order(6)
    @pytest.mark.dependency(
        depends=[
            "TestOneRodBeamTraction::test_end",
            "TestOneRodBeamTorsion::test_end",
            "TestOneRodBeamBendingForce::test_end",
            "TestOneRodBeamBendingMomentum::test_end",
        ]
    )
    def test_random_xyz2xyz(self, ntests=100):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xyz")
            self.direction_force = self.get_random_unit_vector("xyz")
            self.direction_momentum = self.get_random_unit_vector("xyz")
            self.force_value = np.random.uniform(-1, 1)
            self.momentum_value = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(6)
    @pytest.mark.dependency(depends=["TestOneRodBeamAllCharges::test_random_xyz2xyz"])
    def test_end(self):
        pass


@pytest.mark.order(6)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "TestOneRodBeamTraction::test_end",
        "TestOneRodBeamTorsion::test_end",
        "TestOneRodBeamBendingForce::test_end",
        "TestOneRodBeamBendingMomentum::test_end",
        "TestOneRodBeamAllCharges::test_end",
    ]
)
def test_end():
    pass
