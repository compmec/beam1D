import abc
from typing import Optional

import numpy as np

from compmec.strct.__classes__ import ProfileInterface
from compmec.strct.verifytype import PositiveFloat


class ThinProfile(abc.ABC):
    _ratiomax = 0.2
    _ratiodefault = 0.01

    @property
    def e(self) -> PositiveFloat:
        return self.__e

    @e.setter
    def e(self, value: PositiveFloat):
        PositiveFloat.verify(value, "e")
        self.__e = value


class Retangular(ProfileInterface):
    def __init__(self, b: PositiveFloat, h: PositiveFloat):
        self.b = b
        self.h = h

    @property
    def b(self) -> PositiveFloat:
        return self.__b

    @property
    def h(self) -> PositiveFloat:
        return self.__h

    @b.setter
    def b(self, value: PositiveFloat):
        PositiveFloat.verify(value, "b")
        self.__b = value

    @h.setter
    def h(self, value: PositiveFloat):
        PositiveFloat.verify(value, "h")
        self.__h = value

    @property
    def A(self):
        return self.b * self.h


class HollowRetangular(Retangular):
    def __init__(
        self, bi: PositiveFloat, hi: PositiveFloat, be: PositiveFloat, he: PositiveFloat
    ):
        super().__init__(b=(bi + be) / 2, h=(hi + he) / 2)
        if bi >= be:
            raise ValueError("Value of `bi` must be less than `be`")
        if hi >= he:
            raise ValueError("Value of `hi` must be less than `he`")
        self.be = be
        self.he = he
        self.bi = bi
        self.hi = hi
        self.e = min([be - bi, he - hi]) / 2

    @property
    def bi(self) -> PositiveFloat:
        return self.__bi

    @property
    def hi(self) -> PositiveFloat:
        return self.__hi

    @property
    def be(self) -> PositiveFloat:
        return self.__be

    @property
    def he(self) -> PositiveFloat:
        return self.__he

    @bi.setter
    def bi(self, value: PositiveFloat):
        PositiveFloat.verify(value, "bi")
        self.__bi = value

    @hi.setter
    def hi(self, value: PositiveFloat):
        PositiveFloat.verify(value, "hi")
        self.__hi = value

    @be.setter
    def be(self, value: PositiveFloat):
        PositiveFloat.verify(value, "be")
        self.__be = value

    @he.setter
    def he(self, value: PositiveFloat):
        PositiveFloat.verify(value, "he")
        self.__he = value

    @property
    def A(self):
        return self.be * self.he - self.bi * self.hi


class ThinRetangular(HollowRetangular, ThinProfile):
    def __init__(
        self, b: PositiveFloat, h: PositiveFloat, e: Optional[PositiveFloat] = None
    ):
        if e is None:
            e = self._ratiodefault * min([b, h])
        elif e > self._ratiomax * min([b, h]):
            raise ValueError(
                f"The ratio e/min(b, h)={e/min([b,h])} is too big. Use HollowRetangular instead"
            )
        bi, be = b - e, b + e
        hi, he = h - e, h + e
        super().__init__(bi, hi, be, he)


class Square(Retangular):
    def __init__(self, b: PositiveFloat):
        super().__init__(b, b)


class HollowSquare(HollowRetangular):
    def __init__(self, bi: PositiveFloat, be: PositiveFloat):
        super().__init__(bi, bi, be, be)


class ThinSquare(ThinRetangular):
    def __init__(self, b: PositiveFloat, e: Optional[PositiveFloat] = None):
        super().__init__(b, b, e)


class Circle(ProfileInterface):
    def __init__(self, R: PositiveFloat):
        self.R = R

    @property
    def R(self) -> PositiveFloat:
        return self.__R

    @R.setter
    def R(self, value: PositiveFloat) -> None:
        PositiveFloat.verify(value, "R")
        self.__R = value

    @property
    def A(self):
        return np.pi * self.R**2


class HollowCircle(Circle):
    def __init__(self, Ri: PositiveFloat, Re: PositiveFloat):
        R = (Ri + Re) / 2
        super().__init__(R=R)
        self.Ri = Ri
        self.Re = Re
        self.e = self.Re - self.Ri

    @property
    def Ri(self) -> PositiveFloat:
        return self.__Ri

    @property
    def Re(self) -> PositiveFloat:
        return self.__Re

    @property
    def e(self) -> PositiveFloat:
        return self.__e

    @Ri.setter
    def Ri(self, value: PositiveFloat):
        PositiveFloat.verify(value, "Ri")
        self.__Ri = value

    @Re.setter
    def Re(self, value: PositiveFloat):
        PositiveFloat.verify(value, "Re")
        self.__Re = value

    @e.setter
    def e(self, value: PositiveFloat):
        PositiveFloat.verify(value, "e")
        self.__e = value

    @property
    def A(self):
        return np.pi * (self.Re**2 - self.Ri**2)


class ThinCircle(HollowCircle, ThinProfile):
    def __init__(self, R: PositiveFloat, e: Optional[PositiveFloat] = None):
        """
        Creates a thin circle ProfileInterface.
        * R is the mean radius.
        * e is optional thickness.
            If not given, it's 0.01*R
        """
        if e is None:
            e = self._ratiodefault * R
        if e > R * self._ratiomax:
            raise ValueError(
                f"The ratio e/R={e/R} is too big. Use HollowCircle instead"
            )
        Ri = R - 0.5 * e
        Re = R + 0.5 * e
        super().__init__(Ri=Ri, Re=Re)


class ProfileI(ProfileInterface):
    def __init__(
        self,
        b: PositiveFloat,
        h: PositiveFloat,
        t1: PositiveFloat,
        t2: PositiveFloat,
        nu: PositiveFloat,
    ):
        self.b = b
        self.h = h
        self.t1 = t1
        self.t2 = t2
