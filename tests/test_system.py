import pytest

from compmec.strct.element import EulerBernoulli
from compmec.strct.material import Isotropic
from compmec.strct.profile import Circle
from compmec.strct.shower import ShowerStaticSystem
from compmec.strct.system import StaticSystem


@pytest.mark.order(5)
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


class TestStaticSystem:
    @pytest.mark.order(5)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(5)
    @pytest.mark.dependency(depends=["TestStaticSystem::test_begin"])
    def test_main(self):
        system = StaticSystem()
        with pytest.raises(ValueError):
            system.run()
        with pytest.raises(TypeError):
            system.add_element(1)
        with pytest.raises(TypeError):
            system.add_element("asd")
        with pytest.raises(TypeError):
            system.add_element([1, 2, 3])

        beamAB = EulerBernoulli([(0, 0, 0), (1000, 0, 0)])
        system.add_element(beamAB)
        with pytest.raises(TypeError):
            system.add_BC("asd", {"Ux": 0, "Uy": 0})
        with pytest.raises(TypeError):
            system.add_BC(3.4, {"Ux": 0, "Uy": 0})
        with pytest.raises(TypeError):
            system.add_BC((0, 0, 0), "asd")
        with pytest.raises(TypeError):
            system.add_BC((0, 0, 0), {1: 2, 3: 4})
        with pytest.raises(ValueError):
            system.add_BC((0, 0, 0), {"uf": 2, "mk": 4})
        system.add_BC((0, 0, 0), {"Ux": 0, "Uy": 0})

        with pytest.raises(TypeError):
            system.add_dist_load(1, [1.2, 3.4], {3.4: 1})

        with pytest.raises(ValueError):
            system.add_dist_load(beamAB, [1.2, 3.4], {"Ft": (1, 4)})

    @pytest.mark.order(5)
    @pytest.mark.dependency(
        depends=["TestStaticSystem::test_begin", "TestStaticSystem::test_main"]
    )
    def test_end(self):
        pass


@pytest.mark.order(5)
@pytest.mark.dependency(depends=["test_begin", "TestStaticSystem::test_end"])
def test_end():
    pass
