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
    Circle(1)
    Circle(3)
    Circle(2.5)
    with pytest.raises(TypeError):
        Circle("asd")
    with pytest.raises(TypeError):
        Circle("3")
    with pytest.raises(ValueError):
        Circle(-2)
    with pytest.raises(ValueError):
        Circle(0)

    R = np.random.uniform(1, 2)
    A = np.pi * R**2
    circle = Circle(R)
    assert abs(circle.R - R) < TOLERANCE
    assert abs(circle.A - A) < TOLERANCE


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
    assert abs(hollowcircle.R - R) < TOLERANCE
    assert abs(hollowcircle.e - e) < TOLERANCE
    assert abs(hollowcircle.A - A) < TOLERANCE


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_thincircle():
    ThinCircle(1)
    ThinCircle(3)
    with pytest.raises(TypeError):
        ThinCircle("asd")
    with pytest.raises(TypeError):
        ThinCircle("3")
    with pytest.raises(ValueError):
        ThinCircle(-2)
    with pytest.raises(ValueError):
        ThinCircle(0)

    R = np.random.uniform(1, 2)
    thincircle = ThinCircle(R)
    assert abs(thincircle.R - R) < TOLERANCE
    assert thincircle.e < 0.2 * R

    R = np.random.uniform(1, 2)
    e = R * np.random.uniform(0.02, 0.1)
    A = 2 * np.pi * R * e

    thincircle = ThinCircle(R, e)
    assert abs(thincircle.R - R) < TOLERANCE
    assert abs(thincircle.e - e) < TOLERANCE
    assert abs(thincircle.A - A) < TOLERANCE


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_square():
    Square(2)
    Square(2.3)
    with pytest.raises(TypeError):
        Square("asd")
    with pytest.raises(TypeError):
        Square("3")
    with pytest.raises(ValueError):
        Square(-2)
    b = np.random.uniform(1, 2)
    square = Square(b)
    assert abs(square.b - b) < TOLERANCE
    assert abs(square.A - b**2) < TOLERANCE


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_hollowsquare():
    HollowSquare(1, 2)
    HollowSquare(2, 2.3)
    with pytest.raises(TypeError):
        HollowSquare("asd", 3)
    with pytest.raises(TypeError):
        HollowSquare("3", 4)
    with pytest.raises(ValueError):
        HollowSquare(-2, 3)
    with pytest.raises(ValueError):
        HollowSquare(0, 3)
    with pytest.raises(ValueError):
        HollowSquare(3, 3)
    with pytest.raises(ValueError):
        HollowSquare(3, 2)
    bi = np.random.uniform(1, 2)
    be = np.random.uniform(2, 3)
    hollowsquare = HollowSquare(bi, be)
    assert abs(hollowsquare.bi - bi) < TOLERANCE
    assert abs(hollowsquare.be - be) < TOLERANCE
    assert abs(hollowsquare.b - 0.5 * (bi + be)) < TOLERANCE
    assert abs(hollowsquare.hi - bi) < TOLERANCE
    assert abs(hollowsquare.he - be) < TOLERANCE
    assert abs(hollowsquare.h - 0.5 * (bi + be)) < TOLERANCE
    assert abs(hollowsquare.A - be**2 + bi**2) < TOLERANCE


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_thinsquare():
    ThinSquare(1)
    ThinSquare(3)
    with pytest.raises(TypeError):
        ThinSquare("asd")
    with pytest.raises(TypeError):
        ThinSquare("3")
    with pytest.raises(ValueError):
        ThinSquare(-2)
    with pytest.raises(ValueError):
        ThinSquare(0)

    b = np.random.uniform(1, 2)
    thinsquare = ThinSquare(b)
    assert abs(thinsquare.b - b) < TOLERANCE
    assert thinsquare.e < 0.2 * b

    b = np.random.uniform(1, 2)
    e = b * np.random.uniform(0.02, 0.1)
    thinsquare = ThinSquare(b, e)
    assert abs(thinsquare.b - b) < TOLERANCE
    assert abs(thinsquare.e - e) < TOLERANCE
    assert abs(thinsquare.A - 4 * b * e) < TOLERANCE


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_retangular():
    Retangular(1, 2)
    Retangular(7, 4)
    with pytest.raises(TypeError):
        Retangular("asd", 3)
    with pytest.raises(TypeError):
        Retangular("3", 4)
    with pytest.raises(ValueError):
        Retangular(-2, 3)
    with pytest.raises(ValueError):
        Retangular(0, 3)
    with pytest.raises(ValueError):
        Retangular(3, 0)
    b = np.random.uniform(1, 2)
    h = np.random.uniform(2, 3)
    retangular = Retangular(b, h)
    assert abs(retangular.b - b) < TOLERANCE
    assert abs(retangular.h - h) < TOLERANCE


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
    assert abs(hollowretangular.b - 0.5 * (bi + be)) < TOLERANCE
    assert abs(hollowretangular.hi - hi) < TOLERANCE
    assert abs(hollowretangular.he - he) < TOLERANCE
    assert abs(hollowretangular.h - 0.5 * (hi + he)) < TOLERANCE


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_retangular():
    ThinRetangular(1, 2)
    ThinRetangular(7, 4)
    with pytest.raises(TypeError):
        ThinRetangular("asd", 3)
    with pytest.raises(TypeError):
        ThinRetangular("3", 4)
    with pytest.raises(ValueError):
        ThinRetangular(-2, 3)
    with pytest.raises(ValueError):
        ThinRetangular(0, 3)
    with pytest.raises(ValueError):
        ThinRetangular(3, 0)

    b = np.random.uniform(1, 2)
    h = np.random.uniform(1, 2)
    thinretangular = ThinRetangular(b, h)
    assert abs(thinretangular.b - b) < TOLERANCE
    assert abs(thinretangular.h - h) < TOLERANCE

    b = np.random.uniform(1, 2)
    h = np.random.uniform(1, 2)
    e = min(b, h) * np.random.uniform(0.02, 0.1)
    thinretangular = ThinRetangular(b, h, e)
    assert abs(thinretangular.b - b) < TOLERANCE
    assert abs(thinretangular.h - h) < TOLERANCE


@pytest.mark.order(1)
@pytest.mark.dependency(depends=["test_begin"])
def test_end():
    pass
