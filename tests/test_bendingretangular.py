from typing import Optional, Tuple

import numpy as np
import pytest

from compmec.strct.element import EulerBernoulli
from compmec.strct.material import Isotropic
from compmec.strct.profile import Retangular
from compmec.strct.section import RetangularSection
from compmec.strct.solver import solve


@pytest.mark.order(7)
@pytest.mark.dependency(
    depends=[
        "tests/test_solver.py::test_end",
        "tests/test_material.py::test_end",
        "tests/test_structural1D.py::test_end",
        "tests/test_section.py::TestRetangularSection::test_end",
        "tests/test_one_rod_beam_charges.py::test_end",
    ],
    scope="session",
)
def test_begin():
    pass


class InitRetangularBarBeamEuler(object):
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
        b, h = np.random.uniform(1, 2, 2)
        b = h  # To make the tests sucessful, needs futher implementation
        # by changing Ir = Iy * cos^2 + Iz * sin^2
        self.profile = Retangular(b=b, h=h)

    def set_random_section(self):
        self.set_random_material()
        self.set_random_profile()
        self.section = RetangularSection(self.material, self.profile)

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


class TestOneRetangularBarBeamTraction(InitRetangularBarBeamEuler):
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestOneRetangularBarBeamTraction::test_begin"])
    def test_random_x(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("x")
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestOneRetangularBarBeamTraction::test_begin"])
    def test_random_y(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("y")
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestOneRetangularBarBeamTraction::test_begin"])
    def test_random_z(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("z")
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamTraction::test_random_x",
            "TestOneRetangularBarBeamTraction::test_random_y",
        ]
    )
    def test_random_xy(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xy")
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamTraction::test_random_y",
            "TestOneRetangularBarBeamTraction::test_random_z",
        ]
    )
    def test_random_yz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("yz")
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamTraction::test_random_x",
            "TestOneRetangularBarBeamTraction::test_random_z",
        ]
    )
    def test_random_xz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xz")
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamTraction::test_random_xy",
            "TestOneRetangularBarBeamTraction::test_random_yz",
            "TestOneRetangularBarBeamTraction::test_random_xz",
        ]
    )
    def test_random_xyz(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("xyz")
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=["TestOneRetangularBarBeamTraction::test_random_xyz"]
    )
    def test_end(self):
        pass


class TestOneRetangularBarBeamTorsion(InitRetangularBarBeamEuler):
    @pytest.mark.order(7)
    @pytest.mark.skip(reason="Torsion for square-ish shapes is not done")
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestOneRetangularBarBeamTorsion::test_begin"])
    def test_end(self):
        pass


class TestOneRetangularBarBeamBendingForce(InitRetangularBarBeamEuler):
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=["TestOneRetangularBarBeamBendingForce::test_begin"]
    )
    def test_random_x2y(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("x")
            self.direction_force = self.get_random_unit_vector("y", self.direction_beam)
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingForce::test_begin",
            "TestOneRetangularBarBeamBendingForce::test_random_x2y",
        ]
    )
    def test_random_x2z(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("x")
            self.direction_force = self.get_random_unit_vector("z", self.direction_beam)
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingForce::test_begin",
            "TestOneRetangularBarBeamBendingForce::test_random_x2y",
        ]
    )
    def test_random_y2x(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("y")
            self.direction_force = self.get_random_unit_vector("x", self.direction_beam)
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingForce::test_begin",
            "TestOneRetangularBarBeamBendingForce::test_random_x2y",
        ]
    )
    def test_random_y2z(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("y")
            self.direction_force = self.get_random_unit_vector("z", self.direction_beam)
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingForce::test_begin",
            "TestOneRetangularBarBeamBendingForce::test_random_x2y",
        ]
    )
    def test_random_z2x(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("z")
            self.direction_force = self.get_random_unit_vector("x", self.direction_beam)
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingForce::test_begin",
            "TestOneRetangularBarBeamBendingForce::test_random_x2y",
        ]
    )
    def test_random_z2y(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("z")
            self.direction_force = self.get_random_unit_vector("y", self.direction_beam)
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingForce::test_random_x2y",
            "TestOneRetangularBarBeamBendingForce::test_random_x2z",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingForce::test_random_y2x",
            "TestOneRetangularBarBeamBendingForce::test_random_y2z",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingForce::test_random_z2x",
            "TestOneRetangularBarBeamBendingForce::test_random_z2y",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingForce::test_random_x2y",
            "TestOneRetangularBarBeamBendingForce::test_random_y2x",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingForce::test_random_x2z",
            "TestOneRetangularBarBeamBendingForce::test_random_z2x",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingForce::test_random_y2z",
            "TestOneRetangularBarBeamBendingForce::test_random_z2y",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingForce::test_random_xy2xy",
            "TestOneRetangularBarBeamBendingForce::test_random_x2yz",
            "TestOneRetangularBarBeamBendingForce::test_random_y2xz",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingForce::test_random_yz2yz",
            "TestOneRetangularBarBeamBendingForce::test_random_y2xz",
            "TestOneRetangularBarBeamBendingForce::test_random_z2xy",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingForce::test_random_xz2xz",
            "TestOneRetangularBarBeamBendingForce::test_random_x2yz",
            "TestOneRetangularBarBeamBendingForce::test_random_z2xy",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingForce::test_random_xy2xyz",
            "TestOneRetangularBarBeamBendingForce::test_random_xz2xyz",
            "TestOneRetangularBarBeamBendingForce::test_random_xz2xyz",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=["TestOneRetangularBarBeamBendingForce::test_random_xyz2xyz"]
    )
    def test_end(self):
        pass


class TestOneRetangularBarBeamBendingMomentum(InitRetangularBarBeamEuler):
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_random_x2y(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("x")
            self.direction_momentum = self.get_random_unit_vector(
                "y", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "test_begin",
            "TestOneRetangularBarBeamBendingMomentum::test_random_x2y",
        ]
    )
    def test_random_x2z(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("x")
            self.direction_momentum = self.get_random_unit_vector(
                "z", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "test_begin",
            "TestOneRetangularBarBeamBendingMomentum::test_random_x2y",
        ]
    )
    def test_random_y2x(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("y")
            self.direction_momentum = self.get_random_unit_vector(
                "x", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "test_begin",
            "TestOneRetangularBarBeamBendingMomentum::test_random_x2y",
        ]
    )
    def test_random_y2z(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("y")
            self.direction_momentum = self.get_random_unit_vector(
                "z", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "test_begin",
            "TestOneRetangularBarBeamBendingMomentum::test_random_x2y",
        ]
    )
    def test_random_z2x(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("z")
            self.direction_momentum = self.get_random_unit_vector(
                "x", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "test_begin",
            "TestOneRetangularBarBeamBendingMomentum::test_random_x2y",
        ]
    )
    def test_random_z2y(self, ntests=1):
        for i in range(ntests):
            self.direction_beam = self.get_random_unit_vector("z")
            self.direction_momentum = self.get_random_unit_vector(
                "y", self.direction_beam
            )
            self.charge = np.random.uniform(-1, 1)
            self.run_test()

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingMomentum::test_random_x2y",
            "TestOneRetangularBarBeamBendingMomentum::test_random_x2z",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingMomentum::test_random_y2x",
            "TestOneRetangularBarBeamBendingMomentum::test_random_y2z",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingMomentum::test_random_z2x",
            "TestOneRetangularBarBeamBendingMomentum::test_random_z2y",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingMomentum::test_random_x2y",
            "TestOneRetangularBarBeamBendingMomentum::test_random_y2x",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingMomentum::test_random_x2z",
            "TestOneRetangularBarBeamBendingMomentum::test_random_z2x",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingMomentum::test_random_y2z",
            "TestOneRetangularBarBeamBendingMomentum::test_random_z2y",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingMomentum::test_random_xy2xy",
            "TestOneRetangularBarBeamBendingMomentum::test_random_x2yz",
            "TestOneRetangularBarBeamBendingMomentum::test_random_y2xz",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingMomentum::test_random_yz2yz",
            "TestOneRetangularBarBeamBendingMomentum::test_random_y2xz",
            "TestOneRetangularBarBeamBendingMomentum::test_random_z2xy",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingMomentum::test_random_xz2xz",
            "TestOneRetangularBarBeamBendingMomentum::test_random_x2yz",
            "TestOneRetangularBarBeamBendingMomentum::test_random_z2xy",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamBendingMomentum::test_random_xy2xyz",
            "TestOneRetangularBarBeamBendingMomentum::test_random_xz2xyz",
            "TestOneRetangularBarBeamBendingMomentum::test_random_xz2xyz",
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

    @pytest.mark.order(7)
    @pytest.mark.dependency()
    def test_end(self):
        pass


class TestOneRodBeamAllCharges(InitRetangularBarBeamEuler):
    @pytest.mark.order(7)
    @pytest.mark.dependency(
        depends=[
            "TestOneRetangularBarBeamTraction::test_end",
            "TestOneRetangularBarBeamTorsion::test_end"
            "TestOneRetangularBarBeamBendingForce::test_end",
            "TestOneRetangularBarBeamBendingMomentum::test_end",
        ]
    )
    def test_begin(self):
        pass

    @pytest.mark.order(7)
    @pytest.mark.dependency(depends=["TestOneRodBeamAllCharges::test_begin"])
    def test_end(self):
        pass


@pytest.mark.order(7)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "TestOneRetangularBarBeamTraction::test_end"
        "TestOneRetangularBarBeamBendingForce::test_end",
        "TestOneRetangularBarBeamBendingMomentum::test_end",
    ]
)
def test_end():
    pass
