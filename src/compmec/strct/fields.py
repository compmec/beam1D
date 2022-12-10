import compmec.nurbs as nurbs
import numpy as np

from compmec.strct.__classes__ import ComputeFieldBeamInterface, Element1D


class ComputeFieldBeam(ComputeFieldBeamInterface):
    def __init__(self, element: Element1D, result: np.ndarray):
        self.NAME2FUNCTIONS = {
            "u": self.displacement,
            "p": self.position,
            "d": self.deformed,
            "FI": self.internalforce,
            "FE": self.externalforce,
            "MI": self.internalmomentum,
            "ME": self.externalmomentum,
            "TR": self.trescastress,
            "VM": self.vonmisesstress,
        }
        self.element = element
        self.result = result
        degree = 1
        npts = self.result.shape[0]
        self.knotvector = element.path.knotvector

    def __call__(self, fieldname: str) -> nurbs.SplineCurve:
        """
        Computes the field of the element. Fields are:
            ``u``: displacement of the neutral line
            ``p``: position of neutral line (not deformed)
            ``d``: position of neutral line (deformed)
            ``FI``: internal forces on the middle of element
            ``FE``: external forces applied on the element
            ``MI``: internal moment on the middle of the element
            ``ME``: external moment applied on the elmenet
        """
        return self.field(fieldname)

    def field(self, fieldname: str) -> nurbs.SplineCurve:
        keys = list(self.NAME2FUNCTIONS.keys())
        if fieldname not in keys:
            error_msg = (
                f"Received fieldname '{fieldname}' is not valid. They are {keys}"
            )
            raise ValueError(error_msg)
        function = self.NAME2FUNCTIONS[fieldname]
        curve = function()
        return curve

    def displacement(self) -> nurbs.SplineCurve:
        ctrlpts = np.copy(self.result[:, :3])
        curve = nurbs.SplineCurve(self.knotvector, ctrlpts)
        return curve

    def position(self) -> nurbs.SplineCurve:
        return self.element.path

    def deformed(self) -> nurbs.SplineCurve:
        original_position = self.element.path
        displacement = self.field("u")
        return original_position + displacement

    def internalforce(self) -> nurbs.SplineCurve:
        ctrlpts = np.zeros((self.knotvector.npts, 3))
        pairs = np.array([self.element.ts[:-1], self.element.ts[1:]], dtype="float64").T
        for i, (t0, t1) in enumerate(pairs):
            p0, p1 = self.element.path(t0), self.element.path(t1)
            KG = self.element.global_stiffness_matrix(p0, p1)
            UR = self.result[i : i + 2, :]
            FM = np.einsum("ijkl,kl", KG, UR)
            ctrlpts[i, :] = FM[0, :3]
        # Now correct the first value
        t0, t1 = pairs[0]
        p0, p1 = self.element.path(t0), self.element.path(t1)
        KG = self.element.global_stiffness_matrix(p0, p1)
        UR = self.result[0 : 0 + 2, :]
        FM = np.einsum("ijkl,kl", KG, UR)
        ctrlpts[0, :] -= FM[0, :3]
        curve = nurbs.SplineCurve(self.knotvector, ctrlpts)
        return curve

    def externalforce(self) -> nurbs.SplineCurve:
        K = self.element.stiffness_matrix()
        FM = np.einsum("ijkl,kl", K, self.result)
        ctrlpts = FM[:, :3]
        curve = nurbs.SplineCurve(self.knotvector, ctrlpts)
        return curve

    def internalmomentum(self) -> nurbs.SplineCurve:
        ctrlpts = np.zeros((self.knotvector.npts, 3))
        pairs = np.array([self.element.ts[:-1], self.element.ts[1:]], dtype="float64").T
        for i, (t0, t1) in enumerate(pairs):
            p0, p1 = self.element.path(t0), self.element.path(t1)
            KG = self.element.global_stiffness_matrix(p0, p1)
            UR = self.result[i : i + 2, :]
            FM = np.einsum("ijkl,kl", KG, UR)
            ctrlpts[i, :] = FM[0, 3:]
        t0, t1 = pairs[0]
        p0, p1 = self.element.path(t0), self.element.path(t1)
        KG = self.element.global_stiffness_matrix(p0, p1)
        UR = self.result[0:2, :]
        FM = np.einsum("ijkl,kl", KG, UR)
        ctrlpts[0, :] -= FM[0, 3:]
        curve = nurbs.SplineCurve(self.knotvector, ctrlpts)
        return curve

    def externalmomentum(self) -> nurbs.SplineCurve:
        K = self.element.stiffness_matrix()
        FM = np.einsum("ijkl,kl", K, self.result)
        ctrlpts = FM[:, 3:]
        curve = nurbs.SplineCurve(self.knotvector, ctrlpts)
        return curve

    def rotations(self) -> nurbs.SplineCurve:
        raise NotImplementedError

    def trescastress(self) -> nurbs.SplineCurve:
        raise NotImplementedError

    def vonmisesstress(self) -> nurbs.SplineCurve:
        raise NotImplementedError
