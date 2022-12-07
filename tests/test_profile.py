import pytest

from compmec.strct.profile import *

TOLERANCE = 1e-12


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_begin():
    pass


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_circle():
    Circle(5)  # Diameter
    Circle(radius=1)
    Circle(radius=3)
    Circle(radius=2.5)
    Circle(diameter=5)
    with pytest.raises(TypeError):
        Circle(diameter=5, radius=2.5)
    with pytest.raises(TypeError):
        Circle("asd")
    with pytest.raises(TypeError):
        Circle("3")
    with pytest.raises(ValueError):
        Circle(radius=-2)
    with pytest.raises(ValueError):
        Circle(radius=0)

    R = np.random.uniform(1, 2)
    A = np.pi * R**2
    circle = Circle(radius=R)
    assert abs(circle.radius - R) < TOLERANCE
    assert abs(circle.diameter - 2 * R) < TOLERANCE
    assert abs(circle.area - A) < TOLERANCE


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_hollowcircle():
    HollowCircle(1, 2)
    HollowCircle(3, 5)
    with pytest.raises(TypeError):
        HollowCircle(3)
    with pytest.raises(TypeError):
        HollowCircle("asd", 4)
    with pytest.raises(TypeError):
        HollowCircle("3", 4)
    with pytest.raises(TypeError):
        HollowCircle(3, "4")
    with pytest.raises(ValueError):
        HollowCircle(-2, 5)
    with pytest.raises(ValueError):
        HollowCircle(-2, -1)
    with pytest.raises(ValueError):
        HollowCircle(0, 5)
    with pytest.raises(ValueError):
        HollowCircle(0, 0)
    with pytest.raises(ValueError):
        HollowCircle(3, 2)
    with pytest.raises(ValueError):
        HollowCircle(3, 3)

    Ri = np.random.uniform(1, 2)
    Re = np.random.uniform(2, 3)
    A = np.pi * (Re**2 - Ri**2)
    R = 0.5 * (Ri + Re)
    e = Re - Ri
    hollowcircle = HollowCircle(Ri=Ri, Re=Re)
    assert abs(hollowcircle.Ri - Ri) < TOLERANCE
    assert abs(hollowcircle.Re - Re) < TOLERANCE
    assert abs(hollowcircle.radius - R) < TOLERANCE
    assert abs(hollowcircle.thickness - e) < TOLERANCE
    assert abs(hollowcircle.area - A) < TOLERANCE


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_retangular():
    Retangular(1, 2)
    Retangular(7, 4)
    with pytest.raises(TypeError):
        Retangular("asd", 3)
    with pytest.raises(TypeError):
        Retangular("3", 4)
    with pytest.raises(TypeError):
        Retangular(3, "4")
    with pytest.raises(ValueError):
        Retangular(-2, 3)
    with pytest.raises(ValueError):
        Retangular(0, 3)
    with pytest.raises(ValueError):
        Retangular(3, 0)
    b = np.random.uniform(1, 2)
    h = np.random.uniform(2, 3)
    retangular = Retangular(b, h)
    assert abs(retangular.base - b) < TOLERANCE
    assert abs(retangular.height - h) < TOLERANCE


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_hollowretangular():
    HollowRetangular(1, 2, 3, 4)
    HollowRetangular(7, 4, 9, 5)
    with pytest.raises(TypeError):
        HollowRetangular("asd", 3, 1, 1)
    with pytest.raises(TypeError):
        HollowRetangular("3", 4, 5, 5)
    with pytest.raises(ValueError):
        HollowRetangular(-2, 3, 6, 6)
    with pytest.raises(ValueError):
        HollowRetangular(0, 3, 7, 6)
    with pytest.raises(ValueError):
        HollowRetangular(3, 0, 7, 1)
    with pytest.raises(ValueError):
        HollowRetangular(1, 2, 2, 1)
    with pytest.raises(ValueError):
        HollowRetangular(2, 1, 1, 2)
    bi = np.random.uniform(1, 2)
    be = np.random.uniform(2, 3)
    hi = np.random.uniform(1, 2)
    he = np.random.uniform(2, 3)
    hollowretangular = HollowRetangular(bi, hi, be, he)
    assert abs(hollowretangular.bi - bi) < TOLERANCE
    assert abs(hollowretangular.be - be) < TOLERANCE
    assert abs(hollowretangular.base - 0.5 * (bi + be)) < TOLERANCE
    assert abs(hollowretangular.hi - hi) < TOLERANCE
    assert abs(hollowretangular.he - he) < TOLERANCE
    assert abs(hollowretangular.height - 0.5 * (hi + he)) < TOLERANCE
    assert abs(hollowretangular.area - (be * he - bi * hi)) < TOLERANCE


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_Iprofile():
    b, h, t1, t2 = 4, 4, 1, 1
    iprofile = ProfileI(b, h, t1, t2)
    assert abs(iprofile.b - b) < TOLERANCE
    assert abs(iprofile.h - h) < TOLERANCE
    assert abs(iprofile.t1 - t1) < TOLERANCE
    assert abs(iprofile.t2 - t2) < TOLERANCE
    assert iprofile.area > 0


@pytest.mark.order(1)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "test_circle",
        "test_hollowcircle",
        "test_retangular",
        "test_hollowretangular",
    ]
)
def test_end():
    pass
