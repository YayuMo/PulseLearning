from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2 as Sampler


def simulate(qc, shots, backend):
    t_qc = transpile(qc, backend)
    sampler = Sampler(backend)
    sampler.options.default_shots = shots
    result = sampler.run([t_qc]).result()
    dist = result[0].data.meas.get_counts()

    return dist