from compmec.strct.material import Isotropic
import pytest

@pytest.mark.order(1)
@pytest.mark.dependency()
def test_begin():
	pass

@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_main():
	
	Es = [100, 200, 3, 45, 900, 20, 66]
	nus = [0.3, 0.1, 0.05, 0.001, 0.489, 0.45]
	for E, nu in zip(Es,nus):
		G = E/(2*(1+nu))
		K = E*G/(3*(3*G-E))
		L = K - 2*G/3
		mu = G

		mats = []
		mats.append(Isotropic(E=E, nu=nu))
		mats.append(Isotropic(E=E, G=G))
		mats.append(Isotropic(E=E, K=K))
		mats.append(Isotropic(E=E, Lame1=L))
		mats.append(Isotropic(E=E, Lame2=mu))
		mats.append(Isotropic(K=K, nu=nu))
		mats.append(Isotropic(K=K, G=G))
		mats.append(Isotropic(K=K, E=E))
		mats.append(Isotropic(K=K, Lame1=L))
		mats.append(Isotropic(K=K, Lame2=mu))
		mats.append(Isotropic(G=G, nu=nu))
		mats.append(Isotropic(G=G, K=K))
		mats.append(Isotropic(G=G, E=E))
		mats.append(Isotropic(G=G, Lame1=L))
		mats.append(Isotropic(Lame1=L, nu=nu))
		mats.append(Isotropic(Lame1=L, K=K))
		mats.append(Isotropic(Lame1=L, E=E))
		mats.append(Isotropic(Lame1=L, G=G))
		mats.append(Isotropic(Lame1=L, Lame2=mu))
		mats.append(Isotropic(Lame2=mu, nu=nu))
		mats.append(Isotropic(Lame2=mu, K=K))
		mats.append(Isotropic(Lame2=mu, E=E))
		mats.append(Isotropic(Lame2=mu, Lame1=L))

		for i, mat in enumerate(mats):
			assert (mat.E - E) < 1e-6
			assert (mat.G - G) < 1e-6
			assert (mat.K - K) < 1e-6
			assert (mat.nu - nu) < 1e-6
			assert (mat.Lame1 - L) < 1e-6
			assert (mat.Lame2 - G) < 1e-6

@pytest.mark.order(1)
@pytest.mark.dependency(depends=["test_main"])
def test_end():
	pass

if __name__ == "__main__":
	pytest.main()