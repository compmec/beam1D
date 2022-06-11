from beam import Timoshenko
from material import Isotropic
from section import Circle

def main():
	A = (0, 0, 0)  # mm
	B = (1000, 0, 0)  # mm
	C = (500, 500, 0)  # mm
	E = 90000 # MPa
	nu = 0.45  # Poisson coeficient
	steel = Isotropic(E=E, nu=nu)
	circle = Circle(nu=nu, R=1)

	beams = []
	beams.append(Timoshenko(A, B))
	beams.append(Timoshenko(B, C))
	beams.append(Timoshenko(A, C))
	for beam in beams:
		beam.material = steel
		beam.section = circle
		


if __name__ == "__main__":
	main()
