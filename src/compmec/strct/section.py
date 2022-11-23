import numpy as np
from typing import Optional
from compmec.strct.__classes__ import HomogeneousSection



class RetangularSection(HomogeneousSection):

    def shear_coefficient(self):
        nu = self.material.nu
        return 10*(1+nu)/(12+11*nu)

    def compute_areas(self):
        k = self.shear_coefficient()
        self.A[0] = self.profile.A
        self.A[1] = k * self.profile.A
        self.A[2] = k * self.profile.A

    def compute_inertias(self):
        b, h = self.profile.b, self.profile.h
        self.I[1] = b*h**3/12
        self.I[2] = h*b**3/12
        self.I[0] = self.I[1] + self.I[2]
        print("Warning: Inertia for torsional of retangular is not yet defined")
        # raise NotImplementedError("Torsion for a retangular is not defined yet")

class HollowRetangularSection(HomogeneousSection):
    pass

class ThinRetangularSection(HomogeneousSection):
    pass

class SquareSection(HomogeneousSection):
    pass

    def shear_coefficient(self):
        nu = self.material.nu
        return 20*(1+nu)/(4+3*nu)

    def compute_areas(self):
        k = self.shear_coefficient()
        self.A[0] = self.profile.A
        self.A[1] = k * self.A[0]
        self.A[2] = k * self.A[0]
        print("Warning: Areas for a square is not yet well defined")
        # raise NotImplementedError("Areas for a square are not defined")

    def compute_inertias(self):
        print("Warning: Inertias for a square is not yet well defined")
        self.I[1] = self.b**4/12
        self.I[2] = self.I[1]
        self.I[0] = 2*self.I[1]
        # raise NotImplementedError("Inertias for a square are not defined")

class HollowSquareSection(HomogeneousSection):
    pass


class ThinSquareSection(HomogeneousSection):
    pass

class CircleSection(HomogeneousSection):

    def shear_coefficient(self):
        nu = self.material.nu
        return 6*(1+nu)/(7+6*nu)

    def compute_areas(self):
        k = self.shear_coefficient()
        R = self.profile.R
        self.A[0] = self.profile.A
        self.A[1] = k * self.profile.A
        self.A[2] = k * self.profile.A
    
    def compute_inertias(self):
        R4 = self.profile.R**4
        self.I[0] = np.pi * R4 / 2
        self.I[1] = np.pi * R4 / 4
        self.I[2] = np.pi * R4 / 4


class HollowCircle(CircleSection):
    
    def shear_coefficient(self):
        Ri, Re = self.profile.Ri, self.profile.Re
        nu = self.material.nu
        m2 = (Ri/Re)**2
        return 6*(1+nu)/( (7+6*nu) + 4*m2*(5+3*nu)/(1+m2)**2)

    def compute_areas(self):
        k = self.shear_coefficient()
        self.A[0] = self.profile.A
        self.A[1] = k * self.profile.A
        self.A[2] = k * self.profile.A

    def compute_inertias(self):
        Ri4 = self.profile.Ri**4
        Re4 = self.profile.Re**4
        self.I[0] = np.pi * (Re4 - Ri4) / 2
        self.I[1] = np.pi * (Re4 - Ri4) / 4
        self.I[2] = np.pi * (Re4 - Ri4) / 4


class ThinCircle(HollowCircle):
    def shear_coefficient(self):
        nu = self.material.nu
        return 2*(1+nu)/(4+3*nu)
        
    def compute_areas(self):
        k = self.shear_coefficient()
        self.A[0] = self.profile.A
        self.A[1] = k * self.A[0]
        self.A[2] = k * self.A[0]
        
    def compute_inertias(self):
        eR3 = self.profile.e * self.profile.R**3
        self.I[0] = 2 * np.pi * eR3
        self.I[1] = np.pi * eR3
        self.I[2] = np.pi * eR3

class PerfilISection(HomogeneousSection):
    def shear_coefficient(self):
        nu = self.material.nu
        b, h = self.profile.b, self.profile.h
        t1, t2 = self.profile.t1, self.profile.t2
        n = b/h
        m = n*t1/t2
        pt1 = 12+72*m + 150*m**2 + 90*m**3
        pt2 = 11+66*m + 135*m**2 + 90*m**3
        pt3 = 10*n**2 * ((3+nu)*m + 3*m**2)
        numerador = 10*(1+nu)*(1+3*m)**2
        denominador = (pt1 + nu * pt2 + pt3)
        return numerador/denominador

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