from matplotlib import pyplot as plt
from qiskit_dynamics import Solver, DynamicsBackend
from qiskit_dynamics.backend import default_experiment_result_function
from qiskit_dynamics.array import Array
import jax
from qiskit import QuantumCircuit, pulse, transpile, schedule
from qiskit_ibm_runtime.fake_provider.backends.manila.fake_manila import FakeManila

gate_backend = FakeManila()
gate_backend.configuration().hamiltonian['qub']={'0': 2,'1': 2,'2': 2,'3': 2,'4': 2}
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
Array.set_default_backend('jax')
pulse_backend = DynamicsBackend.from_backend(gate_backend)
solver_options = {'method': 'jax_odeint', 'atol': 1e-6, 'rtol': 1e-8}
pulse_backend.set_options(solver_options=solver_options)
pulse_backend.configuration = lambda : gate_backend.configuration()

qc_test = QuantumCircuit(2)
qc_test.h(0)
qc_test.cx(0, 1)
qc_test.measure_all()

t_qc = transpile(qc_test, backend=gate_backend)
pulse_test = schedule(t_qc, backend=gate_backend)

pulse_test.draw()

plt.show()

results = pulse_backend.run(pulse_test).result()
counts = results.get_counts()
print(counts)