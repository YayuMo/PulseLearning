import pennylane as qml
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

# Set to float64 precision and remove jax CPU/GPU warning
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def f1(p, t):
    # polyval(p, t) evaluates a polynomial of degree N=len(p)
    # i.e. p[0]*t**(N-1) + p[1]*t**(N-2) + ... + p[N-2]*t + p[N-1]
    return jnp.polyval(p, t)


def f2(p, t):
    return p[0] * jnp.sin(p[1] * t)


Ht = f1 * qml.PauliX(0) + f2 * qml.PauliY(1)

p1 = jnp.ones(5)
p2 = jnp.array([1.0, jnp.pi])
t = 0.5
print(Ht((p1, p2), t))

coeffs = [1.0] * 2
coeffs += [lambda p, t: jnp.sin(p[0] * t) + jnp.sin(p[1] * t) for _ in range(3)]
ops = [qml.PauliX(i) @ qml.PauliX(i + 1) for i in range(2)]
ops += [qml.PauliZ(i) for i in range(3)]

Ht = qml.dot(coeffs, ops)

# random coefficients
key = jax.random.PRNGKey(777)
subkeys = jax.random.split(key, 3)  # create list of 3 subkeys
params = [jax.random.uniform(subkeys[i], shape=[2], maxval=5) for i in range(3)]
print(Ht(params, 0.5))

ts = jnp.linspace(0.0, 5.0, 100)
fs = Ht.coeffs_parametrized
ops = Ht.ops_parametrized
n_channels = len(fs)
fig, axs = plt.subplots(nrows=n_channels, figsize=(5, 2 * n_channels), gridspec_kw={"hspace": 0})
for n in range(n_channels):
    ax = axs[n]
    ax.plot(ts, fs[n](params[n], ts))
    ax.set_ylabel(f"$f_{n}$")
axs[0].set_title(f"Envelopes $f_i(p_i, t)$ of $\sum_i X_i X_{{i+1}} + \sum_i f_i(p_i, t) Z_i$")
axs[-1].set_xlabel("time t")
plt.tight_layout()
plt.show()

dev = qml.device("default.qubit", range(4))

ts = jnp.array([0.0, 3.0])
H_obj = sum([qml.PauliZ(i) for i in range(4)])


@jax.jit
@qml.qnode(dev, interface="jax")
def qnode(params):
    qml.evolve(Ht)(params, ts)
    return qml.expval(H_obj)


print(qnode(params))

print(jax.grad(qnode)(params))