import abc

import compmec.nurbs as nurbs
import numpy as np

from compmec.strct.__classes__ import ComputeField, Element1D


class ComputeFieldInterface(ComputeField):
    def __init__(self, element: Element1D, ctrlpointsresult: np.ndarray):
        if not isinstance(element, Element1D):
            error_msg = (
                f"First argument must be Element1D! Received type = {type(element)}"
            )
            raise TypeError(error_msg)
        self._element = element
        if len(ctrlpointsresult) != element.path.npts:
            raise ValueError
        knotvector = element.path.knotvector
        self._curveresult = nurbs.SplineCurve(knotvector, ctrlpointsresult)

    @abc.abstractmethod
    def __call__(self, fieldname: str) -> nurbs.SplineCurve:
        raise NotImplementedError

    @abc.abstractmethod
    def _field(self, fieldname: str) -> nurbs.SplineCurve:
        raise NotImplementedError


class ComputeFieldTrussInterface(ComputeFieldInterface):
    pass


class ComputeFieldBeamInterface(ComputeFieldInterface):
    pass


class ComputeFieldBeam(ComputeFieldBeamInterface):
    def __init__(self, element: Element1D, ctrlpointsresult: np.ndarray):
        super().__init__(element, ctrlpointsresult)
        self.NAME2FUNCTIONS = {
            "U": self.displacement,
            "p": self.position,
            "d": self.deformed,
            "FI": self.internalforce,
            "FE": self.externalforce,
            "MI": self.internalmomentum,
            "ME": self.externalmomentum,
            "TR": self.trescastress,
            "VM": self.vonmisesstress,
        }

    def __call__(self, fieldname: str) -> nurbs.SplineCurve:
        """
        Computes the field of the element. Fields are:
            ``U``: displacement of the neutral line
            ``p``: position of neutral line (not deformed)
            ``d``: position of neutral line (deformed)
            ``FI``: internal forces on the middle of element
            ``FE``: external forces applied on the element
            ``MI``: internal moment on the middle of the element
            ``ME``: external moment applied on the elmenet
        """
        return self._field(fieldname)

    def _field(self, fieldname: str) -> nurbs.SplineCurve:
        if fieldname[-1] in ["x", "y", "z", "n", "v", "w"]:
            projection = fieldname[-1]
            fieldname = fieldname[:-1]
        else:
            projection = None
        keys = list(self.NAME2FUNCTIONS.keys())
        if fieldname not in keys:
            error_msg = f"Received fieldname '{fieldname}' is not valid.\n"
            error_msg = f"    With projection '{projection}'\n"
            error_msg += f"They are {keys}"
            raise ValueError(error_msg)
        curve = self.NAME2FUNCTIONS[fieldname]()
        if projection is None:
            return curve
        if projection in ["x", "y", "z"]:
            if projection == "x":
                curve.ctrlpoints = curve.ctrlpoints[:, 0]
            elif projection == "y":
                curve.ctrlpoints = curve.ctrlpoints[:, 1]
            elif projection == "z":
                curve.ctrlpoints = curve.ctrlpoints[:, 2]
            return curve
        raise NotImplementedError

    def displacement(self) -> nurbs.SplineCurve:
        ctrlpoints = np.copy(self._curveresult.ctrlpoints[:, :3])
        knotvector = self._curveresult.knotvector
        curve = self._curveresult.__class__(knotvector, ctrlpoints)
        return curve

    def position(self) -> nurbs.SplineCurve:
        return self._element.path.copy()

    def deformed(self) -> nurbs.SplineCurve:
        original_position = self._element.path
        displacement = self._field("U")
        return original_position + displacement

    def internalforce(self) -> nurbs.SplineCurve:
        resultctrlpoints = self._curveresult.ctrlpoints
        ctrlpts = np.zeros((self._curveresult.npts, 3))
        pairs = np.array(
            [self._element.ts[:-1], self._element.ts[1:]], dtype="float64"
        ).T
        for i, (t0, t1) in enumerate(pairs):
            p0, p1 = self._element.path(t0), self._element.path(t1)
            KG = self._element.global_stiffness_matrix(p0, p1)
            UR = resultctrlpoints[i : i + 2, :]
            FM = np.einsum("ijkl,kl", KG, UR)
            ctrlpts[i, :] = FM[0, :3]
        # Now correct the first value
        t0, t1 = pairs[0]
        p0, p1 = self._element.path(t0), self._element.path(t1)
        KG = self._element.global_stiffness_matrix(p0, p1)
        UR = resultctrlpoints[0 : 0 + 2, :]
        FM = np.einsum("ijkl,kl", KG, UR)
        ctrlpts[0, :] -= FM[0, :3]
        curve = nurbs.SplineCurve(self._curveresult.knotvector, ctrlpts)
        return curve

    def externalforce(self) -> nurbs.SplineCurve:
        K = self._element.stiffness_matrix()
        resultctrlpoints = self._curveresult.ctrlpoints
        FM = np.einsum("ijkl,kl", K, resultctrlpoints)
        ctrlpts = FM[:, :3]
        curve = nurbs.SplineCurve(self._curveresult.knotvector, ctrlpts)
        return curve

    def internalmomentum(self) -> nurbs.SplineCurve:
        resultctrlpoints = self._curveresult.ctrlpoints
        ctrlpts = np.zeros((self._curveresult.npts, 3))
        pairs = np.array(
            [self._element.ts[:-1], self._element.ts[1:]], dtype="float64"
        ).T
        for i, (t0, t1) in enumerate(pairs):
            p0, p1 = self._element.path(t0), self._element.path(t1)
            KG = self._element.global_stiffness_matrix(p0, p1)
            UR = resultctrlpoints[i : i + 2, :]
            FM = np.einsum("ijkl,kl", KG, UR)
            ctrlpts[i, :] = FM[0, 3:]
        t0, t1 = pairs[0]
        p0, p1 = self._element.path(t0), self._element.path(t1)
        KG = self._element.global_stiffness_matrix(p0, p1)
        UR = resultctrlpoints[0:2, :]
        FM = np.einsum("ijkl,kl", KG, UR)
        ctrlpts[0, :] -= FM[0, 3:]
        curve = nurbs.SplineCurve(self._curveresult.knotvector, ctrlpts)
        return curve

    def externalmomentum(self) -> nurbs.SplineCurve:
        K = self._element.stiffness_matrix()
        resultctrlpoints = self._curveresult.ctrlpoints
        FM = np.einsum("ijkl,kl", K, resultctrlpoints)
        ctrlpts = FM[:, 3:]
        curve = nurbs.SplineCurve(self._curveresult.knotvector, ctrlpts)
        return curve

    def rotations(self) -> nurbs.SplineCurve:
        raise NotImplementedError

    def trescastress(self) -> nurbs.SplineCurve:
        raise NotImplementedError

    def vonmisesstress(self) -> nurbs.SplineCurve:
        raise NotImplementedError
