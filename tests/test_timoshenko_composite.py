from typing import Optional, Tuple

import numpy as np
import pytest

from compmec.strct.element import Timoshenko
from compmec.strct.section import GeneralSection


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


@pytest.mark.order(6)
@pytest.mark.dependency(depends=["test_begin"])
def test_creation():
    A = (0, 0, 0)
    B = (1000, 0, 0)
    beamAB = Timoshenko([A, B])
    section = GeneralSection()
    section.A = (10, 10, 10)
    section.I = (1000, 1000, 1000)
    beamAB.section = section


@pytest.mark.order(6)
@pytest.mark.dependency(depends=["test_begin", "test_creation"])
def test_end():
    pass
