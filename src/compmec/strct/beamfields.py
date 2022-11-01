import numpy as np
from compmec.nurbs import SplineBaseFunction, SplineCurve
from compmec.strct.__classes__ import Structural1D


class ComputeField(object):

    __INSTANCE = None

    def __init__(self):
        ComputeField.__INSTANCE = self
        self.__NAME2FUNCTIONS = {"u": self.displacement,
                                 "M": self.momentum,
                                 "F": self.force,
                                 "FE": self.externalforce,
                                 "ME": self.externalmomentum}
    
    @staticmethod
    def getInstance():
        if ComputeField.__INSTANCE is None:
            ComputeField()
        return ComputeField.__INSTANCE
        
    @staticmethod
    def field(fieldname: str, element: Structural1D, result: np.ndarray) -> SplineCurve:
        if not isinstance(element, Structural1D):
            raise TypeError(f"The given element must be a Structural1D instance. Received {type(element)}")
        instance = ComputeField.getInstance()
        if fieldname not in instance.__NAME2FUNCTIONS.keys():
            raise ValueError(f"Received argument is not valid. They are {list(instance.__NAME2FUNCTIONS.keys())}")
        function = instance.__NAME2FUNCTIONS[fieldname]
        curve = function(element, result)
        return curve

    

    def displacement(self, element, result: np.ndarray) -> SplineCurve:
        # if not isinstance(element, EulerBernoulli):
        #     raise TypeError("Element must be a EulerBernoulli instance")
        result = np.array(result)
        if result.ndim != 2:
            raise ValueError("result must be a 2D array")
        npts, dofs = result.shape
        if npts != len(element.ts):
            raise ValueError("The number result.shape[0] must be equal to element.ts")

        U = [0]
        for i, ti in enumerate(element.ts):
            U += [ti]
        U += [1]
        N = SplineBaseFunction(U)
        Ctrlpts = np.copy(result[:, :3])
        curve = SplineCurve(N, Ctrlpts)
        return curve

    def externalforce(self, element, result: np.ndarray) -> SplineCurve:
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

    def externalmomentum(self, element, result: np.ndarray) -> SplineCurve:
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

    def momentum(self, element, result: np.ndarray) -> SplineCurve:
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

    def force(self, element, result: np.ndarray) -> SplineCurve:
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
