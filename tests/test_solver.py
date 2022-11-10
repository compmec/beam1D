from compmec.strct.solver import solve
import pytest
import numpy as np


@pytest.mark.order(1)
@pytest.mark.dependency()
def test_begin():
	pass


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_cantilever2pts():
    TOLERANCE = 1e-6
    Korig = [[[[ 10555.7513,  0,  0,  0, 0,  0],
               [-10555.7513,  0,  0,  0, 0,  0]],
              [[ 0,  0.506676063,  0,  0, 0,  253.338032],
               [ 0, -0.506676063,  0,  0, 0,  253.338032]],
              [[ 0,  0,  0.506676063,  0, -253.338032,  0],
               [ 0,  0, -0.506676063,  0, -253.338032,  0]],
              [[ 0,  0,  0,  32479.2348, 0,  0],
               [ 0,  0,  0, -32479.2348, 0,  0]],
              [[ 0,  0, -253.338032,  0, 168892.021,  0],
               [ 0,  0,  253.338032,  0, 84446.0105,  0]],
              [[ 0,  253.338032,  0,  0, 0,  168892.021],
               [ 0, -253.338032,  0,  0, 0,  84446.0105]]],
             [[[-10555.7513,  0,  0,  0, 0,  0],
               [ 10555.7513,  0,  0, 0, 0,  0]],
              [[ 0, -0.506676063,  0,  0, 0, -253.338032],
               [ 0,  0.506676063,  0,  0, 0, -253.338032]],
              [[ 0,  0, -0.506676063,  0, 253.338032,  0],
               [ 0,  0,  0.506676063,  0, 253.338032,  0]],
              [[ 0,  0,  0, -32479.2348, 0,  0],
               [ 0,  0,  0,  32479.2348, 0,  0]],
              [[ 0,  0, -253.338032,  0, 84446.0105,  0],
               [ 0,  0,  253.338032,  0, 168892.021,  0]],
              [[ 0,  253.338032,  0,  0, 0,  84446.0105],
               [ 0, -253.338032,  0,  0, 0,  168892.021]]]]
    Forig = [[ 0, 0, 0, 0, 0,  0],
             [ 0, -10, 0, 0, 0,  0]]
    Uorig = [[0,0, None, None, None, 0],
             [None, None, None, None, None, None]]
    Korig = np.array(Korig)
    Forig = np.array(Forig)
    Uorig = np.array(Uorig)
    Ugood = [[0,  0, 0, 0, 0, 0],
             [0, -78.9459043, 0, 0, 0, -0.118418856]]
    Fgood = [[0, 10, 0, 0, 0, 10000],
             [0, -10, 0, 0, 0, 0]]
    Utest, Ftest = solve(Korig, Forig, Uorig)
    np.testing.assert_almost_equal(Utest, Ugood, decimal=int(-np.log10(TOLERANCE)))
    np.testing.assert_almost_equal(Ftest, Fgood, decimal=int(-np.log10(TOLERANCE)))


@pytest.mark.order(1)
@pytest.mark.dependency(depends=["test_begin",
                                 "test_cantilever2pts"])
def test_random_notsingular_matrix():
    ntests = 100
    npts = 3
    ndofs = 4
    for i in range(ntests):
        K = np.random.rand(ndofs*npts, ndofs*npts)
        K += K.T
        eigval, eigvec = np.linalg.eigh(K)
        eigval = 100*np.abs(eigval)+50
        K = eigvec @ np.diag(eigval) @ eigvec.T
        U = np.random.rand(ndofs*npts)
        F = K @ U
        K = K.reshape((npts, ndofs, npts, ndofs))
        Ugood = U.reshape((npts, ndofs))
        Fgood = F.reshape((npts, ndofs))
        
        Uorig = np.empty(Ugood.shape, dtype="object")
        for i in range(npts):
            for j in range(ndofs):
                Uorig[i, j] = Ugood[i, j]
        Forig = np.copy(Fgood)

        nunk = np.random.randint(int(0.1*ndofs*npts), int(0.9*ndofs*npts)) # Number of values to set as unknown
        pairsunk = []
        counter = 0
        while counter < nunk:
            pt = np.random.randint(0, npts)
            dof = np.random.randint(0, ndofs)
            pair = (pt, dof)
            if not pair in pairsunk:
                pairsunk.append(pair)
                counter += 1
        for pt in range(npts):
            for dof in range(ndofs):
                if (pt, dof) in pairsunk:
                    Uorig[pt, dof] = None
                else:
                    Forig[pt, dof] = 0
        Utest, Ftest = solve(K, Forig, Uorig)
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)


@pytest.mark.order(1)
@pytest.mark.dependency(depends=["test_begin",
                                 "test_cantilever2pts",
                                 "test_random_notsingular_matrix"])
def test_end():
	pass


def main():
    test_begin()
    test_cantilever2pts()
    test_random_notsingular_matrix()
    test_end()

if __name__ == "__main__":
    main()