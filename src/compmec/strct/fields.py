import numpy as np
import compmec.nurbs as nurbs
from compmec.strct.__classes__ import Structural1DInterface, ComputeFieldBeamInterface

class ComputeFieldBeam(ComputeFieldBeamInterface):
    
    def __init__(self, element: Structural1DInterface, result: np.ndarray):
        self.NAME2FUNCTIONS = {"u": self.displacement,
                               "p": self.position,
                               "d": self.deformed,
                               "FI": self.internalforce,
                               "FE": self.externalforce,
                               "MI": self.internalmomentum,
                               "ME": self.externalmomentum,
                               "TR": self.trescastress,
                               "VM": self.vonmisesstress}
        self.element = element
        self.result = result
        degree = 1
        npts = self.result.shape[0]
        U = [0] * degree + list(np.linspace(0, 1, npts-degree+1)) + [1] * degree
        self.N = nurbs.SplineBaseFunction(U)

    def __call__(self, fieldname: str) -> nurbs.SplineCurve:
        return self.field(fieldname)

    def field(self, fieldname: str) -> nurbs.SplineCurve:
        keys = list(self.NAME2FUNCTIONS.keys())
        if fieldname not in keys:
            raise ValueError(f"Received fieldname '{fieldname}' is not valid. They are {keys}")
        function = self.NAME2FUNCTIONS[fieldname]
        curve = function()
        return curve

    def displacement(self) -> nurbs.SplineCurve:
        Ctrlpts = np.copy(self.result[:, :3])
        curve = nurbs.SplineCurve(self.N, Ctrlpts)
        return curve

    def position(self) -> nurbs.SplineCurve:
        return self.element.path

    def deformed(self) -> nurbs.SplineCurve:
        return self.element.path + self.field("u")

    def internalforce(self) -> nurbs.SplineCurve:
        Ctrlpts = np.zeros((self.N.n, 3))
        pairs = np.array([self.element.ts[:-1], self.element.ts[1:]], dtype="float64").T
        for i, (t0, t1) in enumerate(pairs):
            p0, p1 = self.element.path(t0), self.element.path(t1)
            KG = self.element.global_stiffness_matrix(p0, p1)
            UR = self.result[i:i+2,:]
            FM = np.einsum("ijkl,kl", KG, UR)
            Ctrlpts[i, :] = FM[0, :3]
        Ctrlpts[-1, :] = -FM[-1, :3]
        curve = nurbs.SplineCurve(self.N, Ctrlpts)
        return curve


    def externalforce(self) -> nurbs.SplineCurve:
        K = self.element.stiffness_matrix()
        FM = np.einsum("ijkl,kl", K, self.result)
        Ctrlpts = FM[:, :3]
        curve = nurbs.SplineCurve(self.N, Ctrlpts)
        return curve

    

    def internalmomentum(self) -> nurbs.SplineCurve:
        Ctrlpts = np.zeros((self.N.n, 3))
        pairs = np.array([self.element.ts[:-1], self.element.ts[1:]], dtype="float64").T
        for i, (t0, t1) in enumerate(pairs):
            p0, p1 = self.element.path(t0), self.element.path(t1)
            KG = self.element.global_stiffness_matrix(p0, p1)
            UR = self.result[i:i+2,:]
            FM = np.einsum("ijkl,kl", KG, UR)
            Ctrlpts[i, :] = FM[0, 3:]
        Ctrlpts[-1, :] = -FM[-1, 3:]
        curve = nurbs.SplineCurve(self.N, Ctrlpts)
        return curve

    def externalmomentum(self) -> nurbs.SplineCurve:
        K = self.element.stiffness_matrix()
        FM = np.einsum("ijkl,kl", K, self.result)
        Ctrlpts = FM[:, 3:]
        curve = nurbs.SplineCurve(self.N, Ctrlpts)
        return curve

    
    def rotations(self) -> nurbs.SplineCurve:
        raise NotImplementedError

    def trescastress(self) -> nurbs.SplineCurve:
        raise NotImplementedError

    def vonmisesstress(self) -> nurbs.SplineCurve:
        raise NotImplementedError