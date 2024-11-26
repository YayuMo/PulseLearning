from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
import numpy as np
ultra_simplified_ala_string = "H 0 0 0; H 0 0 0.735"

driver = PySCFDriver(
    atom=ultra_simplified_ala_string,
    basis='sto3g',
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM
)
qmolecule = driver.run()

hamiltonian = qmolecule.hamiltonian
coefficients = hamiltonian.electronic_integrals
second_q_op = hamiltonian.second_q_op()
mapper = ParityMapper()
qubit_op = mapper.map(second_q_op)

# print(qubit_op)
dict = {}
for pauli, coeff in sorted(qubit_op.label_iter()):
    str_info = pauli.__str__()
    dict[str_info] = coeff.real.__str__()

print(dict)