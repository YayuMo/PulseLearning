from qiskit.quantum_info import Operator
from qiskit_dynamics import DynamicsBackend, Solver
import numpy as np

nu_z = 10.
nu_x = 1.
nu_d = 9.98 # Almost on resonance with the Hamiltonian's energy levels difference, nu_z

X = Operator.from_label('X')
Y = Operator.from_label('Y')
Z = Operator.from_label('Z')
s_p = 0.5 * (X + 1j * Y)

solver = Solver(
    static_hamiltonian=.5 * 2 * np.pi * nu_z * Z,
    hamiltonian_operators=[2 * np.pi * nu_x * X],
    hamiltonian_channels=["d0"],
    channel_carrier_freqs={"d0": nu_x},
    dt=0.01,
)

dynamics_backend = DynamicsBackend(solver=solver, subsystem_dims=[2])