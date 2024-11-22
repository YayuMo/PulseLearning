from qiskit.pulse import Schedule, GaussianSquare, Drag, Delay, Play, ControlChannel, DriveChannel

def drag_pulse(backend, amp, angle):
    backend_defaults = backend.defauts()
    inst_sched_map = backend_defaults.instruction_sche_map
    x_pulse = inst_sched_map.get('x', (0)).filter(channels = [DriveChannel(0)], instruction_type=[Play]).instructions[0][1].pulse
    duration_parameter = x_pulse.parameters['duration']
    sigma_parameter = x_pulse.parameters['sigma']
    beta_parameter = x_pulse.parameters['beta']
    pulse1 = Drag(duration=duration_parameter, sigma=sigma_parameter, beta=beta_parameter, amp=amp, angle=angle)
    return pulse1
