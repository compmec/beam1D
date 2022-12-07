from typing import Optional

import numpy as np

from compmec.strct.__classes__ import HomogeneousSection, Material, Profile
from compmec.strct.profile import *


class RetangularSection(HomogeneousSection):
    def shear_coefficient(self):
        nu = self.material.nu
        return 10 * (1 + nu) / (12 + 11 * nu)

    def compute_areas(self):
        k = self.shear_coefficient()
        A = np.zeros(3, dtype="float64")
        A[0] = self.profile.area
        A[1] = k * self.profile.area
        A[2] = k * self.profile.area
        self.A = A

    def compute_inertias(self):
        b, h = self.profile.base, self.profile.height
        I = np.zeros(3, dtype="float64")
        I[1] = b * h**3 / 12
        I[2] = h * b**3 / 12
        I[0] = I[1] + I[2]
        print("Warning: Inertia for torsional of retangular is not yet defined")
        self.I = I
        # raise NotImplementedError("Torsion for a retangular is not defined yet")


class HollowRetangularSection(HomogeneousSection):
    pass


class CircleSection(HomogeneousSection):
    def shear_coefficient(self):
        nu = self.material.nu
        return 6 * (1 + nu) / (7 + 6 * nu)

    def compute_areas(self):
        k = self.shear_coefficient()
        A = np.zeros(3, dtype="float64")
        A[0] = self.profile.area
        A[1] = k * self.profile.area
        A[2] = k * self.profile.area
        self.A = A

    def compute_inertias(self):
        R4 = self.profile.radius**4
        I = np.zeros(3, dtype="float64")
        I[0] = np.pi * R4 / 2
        I[1] = np.pi * R4 / 4
        I[2] = np.pi * R4 / 4
        self.I = I


class HollowCircleSection(CircleSection):
    def shear_coefficient(self):
        Ri, Re = self.profile.Ri, self.profile.Re
        nu = self.material.nu
        m2 = (Ri / Re) ** 2
        return 6 * (1 + nu) / ((7 + 6 * nu) + 4 * m2 * (5 + 3 * nu) / (1 + m2) ** 2)

    def compute_areas(self):
        k = self.shear_coefficient()
        A = np.zeros(3, dtype="float64")
        A[0] = self.profile.area
        A[1] = k * self.profile.area
        A[2] = k * self.profile.area
        self.A = A

    def compute_inertias(self):
        Ri4 = self.profile.Ri**4
        Re4 = self.profile.Re**4
        I = np.zeros(3, dtype="float64")
        I[0] = np.pi * (Re4 - Ri4) / 2
        I[1] = np.pi * (Re4 - Ri4) / 4
        I[2] = np.pi * (Re4 - Ri4) / 4
        self.I = I


class PerfilISection(HomogeneousSection):
    def shear_coefficient(self):
        nu = self.material.nu
        b, h = self.profile.base, self.profile.height
        t1, t2 = self.profile.t1, self.profile.t2
        n = b / h
        m = n * t1 / t2
        pt1 = 12 + 72 * m + 150 * m**2 + 90 * m**3
        pt2 = 11 + 66 * m + 135 * m**2 + 90 * m**3
        pt3 = 10 * n**2 * ((3 + nu) * m + 3 * m**2)
        numerador = 10 * (1 + nu) * (1 + 3 * m) ** 2
        denominador = pt1 + nu * pt2 + pt3
        return numerador / denominador


class GeneralSection(HomogeneousSection):
    def __init__(self, curves: list, nu: float):
        """
        curves is a list of closed curves that defines the geometry
        Each curve is a Nurbs, with the points.
        It's possible to have a circle, only with one curve, a circle
        Until now, it's not implemented
        """
        super().__init__(nu)
        raise Exception("Not implemented")


def create_section_from_material_profile(
    material: Material, profile: Profile
) -> HomogeneousSection:
    if not isinstance(material, Material):
        raise TypeError
    if not isinstance(profile, Profile):
        raise TypeError
    mapto = {
        Retangular: RetangularSection,
        HollowRetangular: HollowRetangularSection,
        Circle: CircleSection,
        HollowCircle: HollowCircleSection,
    }
    for profileclass, sectionclass in mapto.items():
        if type(profile) == profileclass:
            return sectionclass(material, profile)
    raise ValueError(f"Could not translate profile {type(profile)} to a section")
