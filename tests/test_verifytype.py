import pytest

from compmec.strct.verifytype import *


@type_check
def funct1good() -> None:
    return None


@type_check
def funct2(value: Float) -> None:
    return None


@type_check
def funct3(value: Float) -> Float:
    return value**2


@type_check
def funct4(value: PositiveFloat) -> Float:
    return 1


@type_check
def funct5(value: Float) -> PositiveFloat:
    return value


@type_check
def funct5(value: Float) -> PositiveFloat:
    return value


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_begin():
    pass


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_main():
    funct2(1.1)
    with pytest.raises(TypeError):
        funct2("1")
    funct3(1.1)
    with pytest.raises(TypeError):
        funct3("2")
    funct4(1.1)
    with pytest.raises(ValueError):
        funct4(-1)
    with pytest.raises(ValueError):
        funct5(-1)
    with pytest.raises(TypeError):
        funct5("asd")


@pytest.mark.order(1)
@pytest.mark.dependency(depends=["test_begin", "test_main"])
def test_end():
    pass
