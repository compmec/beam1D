import numpy as np
from compmec.nurbs import SplineBaseFunction, SplineCurve
from compmec.strct.__classes__ import Structural1D, ComputeFieldBeamInterface

class ComputeFieldBeam(ComputeFieldBeamInterface):
    
    def __init__(self, element: Structural1D, result: np.ndarray):
        self.NAME2FUNCTIONS = {"u": self.displacement,
                               "p": self.position,
                               "d": self.deformed,
                               "MI": self.internalmomentum,
                               "FI": self.internalforce,
                               "FE": self.externalforce,
                               "ME": self.externalmomentum}
        self.element = element
        self.result = result

    def field(self, fieldname: str) -> SplineCurve:
        if fieldname not in self.NAME2FUNCTIONS.keys():
            raise ValueError(f"Received fieldname '{fieldname}' is not valid. They are {list(ComputeFieldBeam.__NAME2FUNCTIONS.keys())}")
        return self.NAME2FUNCTIONS[fieldname]()

    def displacement(self) -> SplineCurve:
        

        U = [0]
        for i, ti in enumerate(element.ts):
            U += [ti]
        U += [1]
        N = SplineBaseFunction(U)
        Ctrlpts = np.copy(result[:, :3])
        curve = SplineCurve(N, Ctrlpts)
        return curve

    def position(self) -> SplineCurve:
        return self.element.path

    def deformed(self) -> SplineCurve:
        return self.element.path + self.field("u")

    def externalforce(self) -> SplineCurve:
        result = np.array(result)
        if result.ndim != 2:
            raise ValueError("result must be a 2D array")
        npts, dofs = result.shape
        if npts != len(element.ts):
            raise ValueError("The number result.shape[0] must be equal to element.ts")

        U = [0]
        U += list(element.ts)
        U += [1]
        N = SplineBaseFunction(U)
        K = element.stiffness_matrix()
        FM = np.einsum("ijkl,kl", K, result)
        Ctrlpts = FM[:, :3]
        curve = SplineCurve(N, Ctrlpts)
        return curve

    def externalmomentum(self) -> SplineCurve:
        result = np.array(result)
        if result.ndim != 2:
            raise ValueError("result must be a 2D array")
        npts, dofs = result.shape
        if npts != len(element.ts):
            raise ValueError("The number result.shape[0] must be equal to element.ts")

        U = [0]
        U += list(element.ts)
        U += [1]
        N = SplineBaseFunction(U)
        K = element.stiffness_matrix()
        FM = np.einsum("ijkl,kl", K, result)
        Ctrlpts = FM[:, 3:]
        curve = SplineCurve(N, Ctrlpts)
        return curve

    def internalmomentum(self) -> SplineCurve:
        result = np.array(result)
        if result.ndim != 2:
            raise ValueError("result must be a 2D array")
        npts, dofs = result.shape
        if npts != len(element.ts):
            raise ValueError("The number result.shape[0] must be equal to element.ts")

        U = [0]
        U += list(element.ts)
        U += [1]
        N = SplineBaseFunction(U)
        Ctrlpts = np.zeros((N.n, 3))
        pairs = np.array([element.ts[:-1], element.ts[1:]]).T
        for i, (t0, t1) in enumerate(pairs):
            p0, p1 = element.path(t0), element.path(t1)
            KG = element.global_stiffness_matrix(p0, p1)
            UR = result[i:i+2,:]
            FM = np.einsum("ijkl,kl", KG, UR)
            Ctrlpts[i, :] = FM[0, 3:]
        Ctrlpts[-1, :] = -FM[-1, 3:]
        curve = SplineCurve(N, Ctrlpts)
        return curve

    def internalforce(self) -> SplineCurve:
        result = np.array(result)
        if result.ndim != 2:
            raise ValueError("result must be a 2D array")
        npts, dofs = result.shape
        if npts != len(element.ts):
            raise ValueError("The number result.shape[0] must be equal to element.ts")

        U = [0]
        U += list(element.ts)
        U += [1]
        N = SplineBaseFunction(U)
        Ctrlpts = np.zeros((N.n, 3))
        pairs = np.array([element.ts[:-1], element.ts[1:]]).T
        for i, (t0, t1) in enumerate(pairs):
            p0, p1 = element.path(t0), element.path(t1)
            KG = element.global_stiffness_matrix(p0, p1)
            UR = result[i:i+2,:]
            FM = np.einsum("ijkl,kl", KG, UR)
            Ctrlpts[i, :] = FM[0, :3]
        Ctrlpts[-1, :] = -FM[-1, :3]
        curve = SplineCurve(N, Ctrlpts)
        return curve
