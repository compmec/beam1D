import numpy as np
import pytest

from compmec.strct.geometry import Geometry1D, Point2D, Point3D


@pytest.mark.order(2)
@pytest.mark.dependency()
def test_begin():
    pass


class TestPoint2D:
    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(2)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestPoint2D::test_begin"])
    def test_creation(self):
        Point2D([3, 4])

    @pytest.mark.order(2)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(
        depends=["TestPoint2D::test_begin", "TestPoint2D::test_creation"]
    )
    def test_comparation(self):
        A = Point2D([4, 5])
        B = Point2D([-3, 2.2])
        C = Point2D([4.0, 5.0])
        assert A == A
        assert A != B
        assert A == C
        assert A == (4, 5)
        assert B == (-3, 2.2)
        assert B == (-3.0, 2.2)

    @pytest.mark.order(2)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(
        depends=["TestPoint2D::test_begin", "TestPoint2D::test_comparation"]
    )
    def test_add_sub(self):
        A = Point2D([4, 5])
        B = Point2D([-3, 2.2])
        D = Point2D([1, 7.2])  # A+B
        E = Point2D([7.0, 2.8])  # A-B
        assert A + B == D
        assert A - B == E
        assert A - (4, 5) == (0, 0)
        assert A + (4, 5) == (8, 10)
        assert (4, 5) - A == (0, 0)
        assert (4, 5) + A == (8, 10)

    @pytest.mark.order(2)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(
        depends=[
            "TestPoint2D::test_begin",
            "TestPoint2D::test_creation",
            "TestPoint2D::test_comparation",
            "TestPoint2D::test_add_sub",
        ]
    )
    def test_fail(self):
        with pytest.raises(TypeError):
            Point2D("asd")
        with pytest.raises(ValueError):
            Point2D([1, 2, 3, 4])
        with pytest.raises(TypeError):
            Point2D([1, "4"])
        with pytest.raises(TypeError):
            Point2D(["asd", {1: 2}])
        A = Point2D([3, 5])
        assert A != 1
        assert A != "asd"
        assert A != [3, 2]

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=[
            "TestPoint2D::test_begin",
            "TestPoint2D::test_add_sub",
            "TestPoint2D::test_fail",
        ]
    )
    def test_end(self):
        pass


class TestPoint3D:
    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["test_begin"])
    def test_begin(self):
        pass

    @pytest.mark.order(2)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(depends=["TestPoint3D::test_begin"])
    def test_creation(self):
        Point3D([3, 4, 5])

    @pytest.mark.order(2)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(
        depends=["TestPoint3D::test_begin", "TestPoint3D::test_creation"]
    )
    def test_comparation(self):
        A = Point3D([4, 5, 3.2])
        B = Point3D([-3, 2.2, 9])
        C = Point3D([4.0, 5.0, 3.2])
        assert A == A
        assert A != B
        assert A == C
        assert A == (4, 5, 3.2)
        assert B == (-3, 2.2, 9)
        assert B == (-3.0, 2.2, 9.0)

    @pytest.mark.order(2)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(
        depends=["TestPoint3D::test_begin", "TestPoint3D::test_comparation"]
    )
    def test_add_sub(self):
        A = Point3D([4, 5, 3.2])
        B = Point3D([-3, 2.2, 9])
        D = Point3D([1, 7.2, 12.2])  # A+B
        E = Point3D([7.0, 2.8, -5.8])  # A-B
        assert A + B == D
        assert A - B == E
        assert A - (4, 5, 3.2) == (0, 0, 0)
        assert A + (4, 5, 3.2) == (8, 10, 6.4)
        assert (4, 5, 3.2) - A == (0, 0, 0)
        assert (4, 5, 3.2) + A == (8, 10, 6.4)

    @pytest.mark.order(2)
    @pytest.mark.timeout(1)
    @pytest.mark.dependency(
        depends=[
            "TestPoint3D::test_begin",
            "TestPoint3D::test_creation",
            "TestPoint3D::test_comparation",
            "TestPoint3D::test_add_sub",
        ]
    )
    def test_fail(self):
        with pytest.raises(TypeError):
            Point3D("asd")
        with pytest.raises(ValueError):
            Point3D([1, 2, 3, 4])
        with pytest.raises(TypeError):
            Point3D([1, 2, "4"])
        with pytest.raises(TypeError):
            Point3D(["asd", 3, {1: 2}])
        A = Point3D([3, 4, 5])
        assert A != 1
        assert A != "asd"
        assert A != [3, 2]

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=[
            "TestPoint3D::test_begin",
            "TestPoint3D::test_add_sub",
            "TestPoint3D::test_fail",
        ]
    )
    def test_end(self):
        pass


class TestGeometry:
    @pytest.mark.order(2)
    @pytest.mark.dependency(depends=["test_begin", "TestPoint3D::test_end"])
    def test_begin(self):
        pass

    @pytest.mark.order(2)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(depends=["TestGeometry::test_begin"])
    def test_creation(self):
        geometry = Geometry1D()

    @pytest.mark.order(2)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(depends=["TestGeometry::test_creation"])
    def test_create_point(self):
        geometry = Geometry1D()
        geometry.create_point([2, 3, 4])

    @pytest.mark.order(2)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(depends=["TestGeometry::test_create_point"])
    def test_find_point(self):
        geometry = Geometry1D()
        geometry.create_point([2, 3, 4])
        geometry.create_point([-2, 3.0, 5])
        assert geometry.find_point([2, 3, 4]) == 0
        assert geometry.find_point([-2, 3.0, 5]) == 1
        assert geometry.find_point([0, 0, 0]) is None
        assert geometry.find_point([0, 0, 0], tolerance=6) == 0
        assert geometry.find_point([0, 0, 0], tolerance=5.5) == 0
        assert geometry.find_point([0, 0, 0], tolerance=5) is None
        assert geometry.npts == 2
        all_points = ((2, 3, 4), (-2, 3.0, 5))
        np.testing.assert_almost_equal(geometry.points, all_points)

    @pytest.mark.order(2)
    @pytest.mark.timeout(2)
    @pytest.mark.dependency(
        depends=["TestGeometry::test_create_point", "TestGeometry::test_find_point"]
    )
    def test_fail(self):
        geometry = Geometry1D()
        geometry.create_point([2, 3, 4])
        geometry.create_point([4, 3, 4])
        with pytest.raises(TypeError):
            geometry.find_point([2, 3, 4], tolerance="asd")
        with pytest.raises(ValueError):
            geometry.find_point([2, 3, 4], tolerance=-1)
        with pytest.raises(ValueError):
            geometry.find_point([2, 3, 4], tolerance=0)
        with pytest.raises(ValueError):
            geometry.create_point([2, 3, 4])
        with pytest.raises(ValueError):
            geometry.find_point([3, 3, 4], tolerance=2)

    @pytest.mark.order(2)
    @pytest.mark.dependency(
        depends=["TestGeometry::test_find_point", "TestGeometry::test_fail"]
    )
    def test_end(self):
        pass


@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(
    depends=[
        "test_begin",
        "TestPoint2D::test_end",
        "TestPoint3D::test_end",
        "TestGeometry::test_end",
    ]
)
def test_end():
    pass
