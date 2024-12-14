import pennylane as qml
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

# Set to float64 precision and remove jax CPU/GPU warning
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

data = qml.data.load("qchem", molname="HeH+", basis="STO-3G", bondlength=1.5)[0]
H_obj = data.tapered_hamiltonian
H_obj_coeffs, H_obj_ops = H_obj.terms()

# casting the Hamiltonian coefficients to a jax Array
H_obj = qml.Hamiltonian(jnp.array(H_obj_coeffs), H_obj_ops)
E_exact = data.fci_energy
n_wires = len(H_obj.wires)

def a(wires):
    return 0.5 * qml.PauliX(wires) + 0.5j * qml.PauliY(wires)


def ad(wires):
    return 0.5 * qml.PauliX(wires) - 0.5j * qml.PauliY(wires)


omega = 2 * jnp.pi * jnp.array([4.8080, 4.8333])
g = 2 * jnp.pi * jnp.array([0.01831, 0.02131])

H_D = qml.dot(omega, [ad(i) @ a(i) for i in range(n_wires)])
H_D += qml.dot(
    g,
    [ad(i) @ a((i + 1) % n_wires) + ad((i + 1) % n_wires) @ a(i) for i in range(n_wires)],
)

def normalize(x):
    """Differentiable normalization to +/- 1 outputs (shifted sigmoid)"""
    return (1 - jnp.exp(-x)) / (1 + jnp.exp(-x))

# Because ParametrizedHamiltonian expects each callable function to have the signature
# f(p, t) but we have additional parameters it depends on, we create a wrapper function
# that constructs the callables with the appropriate parameters imprinted on them
def drive_field(T, omega, sign=1.0):
    def wrapped(p, t):
        # The first len(p)-1 values of the trainable params p characterize the pwc function
        amp = qml.pulse.pwc(T)(p[:-1], t)
        # The amplitude is normalized to maximally reach +/-20MHz (0.02GHz)
        amp = 0.02 * normalize(amp)

        # The last value of the trainable params p provides the drive frequency deviation
        # We normalize as the difference to drive can maximally be +/-1 GHz
        # d_angle = normalize(p[-1])
        # phase = jnp.exp(sign * 1j * (omega + d_angle) * t)
        phase = jnp.exp(sign * 1j * omega * t)
        return amp * phase

    return wrapped

duration = 15.0

fs = [drive_field(duration, omega[i], 1.0) for i in range(n_wires)]
fs += [drive_field(duration, omega[i], -1.0) for i in range(n_wires)]
ops = [a(i) for i in range(n_wires)]
ops += [ad(i) for i in range(n_wires)]

H_C = qml.dot(fs, ops)

