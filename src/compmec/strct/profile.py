from math import isnan, nan

import numpy as np

from compmec.strct.__classes__ import Profile
from compmec.strct.verifytype import PositiveFloat


class Retangular(Profile):
    def __init__(self, base: PositiveFloat, height: PositiveFloat):
        PositiveFloat.verify(base, "base")
        PositiveFloat.verify(height, "height")
        self._base = base
        self._height = height

    @property
    def base(self) -> PositiveFloat:
        return self._base

    @property
    def height(self) -> PositiveFloat:
        return self._height

    @property
    def area(self) -> PositiveFloat:
        return self.base * self.height


class HollowRetangular(Retangular):
    def __init__(
        self, bi: PositiveFloat, hi: PositiveFloat, be: PositiveFloat, he: PositiveFloat
    ):
        PositiveFloat.verify(bi, "bi")
        PositiveFloat.verify(be, "be")
        PositiveFloat.verify(hi, "hi")
        PositiveFloat.verify(he, "he")
        super().__init__(base=(bi + be) / 2, height=(hi + he) / 2)
        if bi >= be:
            raise ValueError("Value of `bi` must be less than `be`")
        if hi >= he:
            raise ValueError("Value of `hi` must be less than `he`")
        self._be = be
        self._he = he
        self._bi = bi
        self._hi = hi

    @property
    def bi(self) -> PositiveFloat:
        return self._bi

    @property
    def hi(self) -> PositiveFloat:
        return self._hi

    @property
    def be(self) -> PositiveFloat:
        return self._be

    @property
    def he(self) -> PositiveFloat:
        return self._he

    @property
    def area(self) -> PositiveFloat:
        return self.be * self.he - self.bi * self.hi


class Circle(Profile):
    def __init__(
        self, diameter: PositiveFloat = nan, radius: PositiveFloat = nan
    ) -> None:
        """Takes either a `radius` or a `diameter` but not both."""
        if not isnan(radius) ^ isnan(diameter):
            raise TypeError("Either radius or diameter required")
        if isnan(diameter):
            PositiveFloat.verify(radius, "radius")
            self._radius = radius
        else:
            PositiveFloat.verify(diameter, "diameter")
            self._radius = diameter / 2

    @property
    def radius(self) -> PositiveFloat:
        return self._radius

    @property
    def diameter(self) -> PositiveFloat:
        return 2 * self._radius

    @property
    def area(self) -> PositiveFloat:
        return np.pi * self.radius**2


class HollowCircle(Circle):
    def __init__(self, Ri: PositiveFloat, Re: PositiveFloat):
        PositiveFloat.verify(Ri, "Ri")
        PositiveFloat.verify(Re, "Re")
        radius = (Ri + Re) / 2
        super().__init__(radius=radius)
        if Ri >= Re:
            error_msg = (
                f"The internal radius Ri = {Ri:.2f} must be less than Re = {Re:.2f}"
            )
            raise ValueError(error_msg)
        self.__Ri = Ri
        self.__Re = Re
        self.__e = self.Re - self.Ri

    @property
    def Ri(self) -> PositiveFloat:
        return self.__Ri

    @property
    def Re(self) -> PositiveFloat:
        return self.__Re

    @property
    def thickness(self) -> PositiveFloat:
        return self.__e

    @property
    def area(self) -> PositiveFloat:
        return np.pi * (self.Re**2 - self.Ri**2)


class PerfilI(Profile):
    """
    https://engineering.stackexchange.com/questions/42689/shear-coefficient-for-an-i-beam
    height h (z axis)
    width w (y axis)
    flange thickness t
    web thickness s
    """

    def __init__(
        self,
        b: PositiveFloat,
        h: PositiveFloat,
        t: PositiveFloat,
        s: PositiveFloat,
    ):
        PositiveFloat.verify(b, "b")
        PositiveFloat.verify(h, "h")
        PositiveFloat.verify(t, "t")
        PositiveFloat.verify(s, "s")
        self._b = b
        self._h = h
        self._t = t
        self._s = s

    @property
    def b(self) -> PositiveFloat:
        return self._b

    @property
    def h(self) -> PositiveFloat:
        return self._h

    @property
    def t(self) -> PositiveFloat:
        return self._t

    @property
    def s(self) -> PositiveFloat:
        return self._s

    @property
    def area(self) -> PositiveFloat:
        return 2 * self.b * self.t + self.s * (self.h - self.t)
