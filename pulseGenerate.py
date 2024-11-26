from qiskit import pulse
from qiskit.pulse import Schedule, GaussianSquare, Drag, Delay, Play, ControlChannel, DriveChannel
import copy
from scipy.optimize import minimize, LinearConstraint
import numpy as np

def drag_pulse(backend, amp, angle):
    backend_defaults = backend.defauts()
    inst_sched_map = backend_defaults.instruction_sche_map
    x_pulse = inst_sched_map.get('x', (0)).filter(channels = [DriveChannel(0)], instruction_type=[Play]).instructions[0][1].pulse
    duration_parameter = x_pulse.parameters['duration']
    sigma_parameter = x_pulse.parameters['sigma']
    beta_parameter = x_pulse.parameters['beta']
    pulse1 = Drag(duration=duration_parameter, sigma=sigma_parameter, beta=beta_parameter, amp=amp, angle=angle)
    return pulse1

def cr_pulse(backend, amp, angle, duration):
  backend_defaults = backend.defaults()
  inst_sched_map = backend_defaults.instruction_schedule_map
  cr_pulse = inst_sched_map.get('cx', (0, 1)).filter(channels = [ControlChannel(0)], instruction_types=[Play]).instructions[0][1].pulse
  cr_params = {}
  cr_params['duration'] = cr_pulse.parameters['duration']
  cr_params['amp'] = cr_pulse.parameters['amp']
  cr_params['angle'] = cr_pulse.parameters['angle']
  cr_params['sigma'] = cr_pulse.parameters['sigma']
  cr_params['width'] = cr_pulse.parameters['width']
  cr_risefall = (cr_params['duration'] - cr_params['width']) / (2 * cr_params['sigma'])
  angle_parameter = angle
  duration_parameter =  duration
  sigma_parameter = cr_pulse.parameters['sigma']
  width_parameter = int(duration_parameter - 2 * cr_risefall * cr_params['sigma'])
  #declare pulse parameters and build GaussianSquare pulse
  pulse1 = GaussianSquare(duration = duration_parameter, amp = amp, angle = angle_parameter, sigma = sigma_parameter, width=width_parameter)
  return pulse1

def HE_pulse(backend, amp, angle, width):
    with pulse.build(backend) as my_program1:
      # layer 1
      sched_list = []
      with pulse.build(backend) as sched1:
          qubits = (0,1,2,3)
          for i in range(4):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(qubits[i]))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2:
          uchan = pulse.control_channels(0, 1)[0]
          pulse.play(cr_pulse(backend,amp[4], angle[4], width[0]), uchan)
      sched_list.append(sched2)


      with pulse.build(backend) as sched4:
          uchan = pulse.control_channels(1, 2)[0]
          pulse.play(cr_pulse(backend, amp[5], angle[5], width[1]), uchan)
      sched_list.append(sched4)

      with pulse.build(backend) as sched6:
          uchan = pulse.control_channels(2,3)[0]
          pulse.play(cr_pulse(backend, amp[6],angle[6], width[2]), uchan)
      sched_list.append(sched6)


      with pulse.build(backend) as my_program:
        with pulse.transpiler_settings(initial_layout= [0,1,2,3]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)

    return my_program

def measurement_pauli(prepulse, pauli_string, backend, n_qubit):
    with pulse.build(backend) as pulse_measure:
        pulse.call(copy.deepcopy(prepulse))
        for ind,pauli in enumerate(pauli_string):
            if(pauli=='X'):
                pulse.u2(0, np.pi, ind)
            if(pauli=='Y'):
                pulse.u2(0, np.pi/2, ind)
        for qubit in range(n_qubit):
            pulse.barrier(qubit)
        pulse.measure(range(n_qubit))
    return pulse_measure

def n_one(bitstring, key):
    results = 0
    for ind,b in enumerate(reversed(bitstring)):
        if((b=='1')&(key[ind]!='I')):
            results+=1
    return results

def expectation_value(counts,shots,key):
    results = 0
    for bitstring in counts:
        if(n_one(bitstring, key)%2==1):
            results -= counts[bitstring]/shots
        else:
            results += counts[bitstring]/shots
    return results

def run_pulse_sim(meas_pulse, key, pulse_backend, backend, n_shot):
    results = pulse_backend.run(meas_pulse).result()
    counts = results.get_counts()
    expectation = expectation_value(counts,n_shot,key)
    return expectation

def gen_LC_vqe(parameters):
    lb = np.zeros(parameters)
    ub = np.ones(parameters)
    LC = (LinearConstraint(np.eye(parameters),lb,ub,keep_feasible=False))
    return LC

def vqe_one(prepulse,n_qubit,n_shot,pulse_backend, backend,key,value):
    all_Is = True
    for key_ele in key:
        if(key_ele!='I'):
            all_Is = False
    if(all_Is):
        return value
    meas_pulse = measurement_pauli(prepulse=prepulse, pauli_string=key, backend=backend, n_qubit=n_qubit)
    return value*run_pulse_sim(meas_pulse, key, pulse_backend, backend, n_shot)

def vqe(params,pauli_dict,pulse_backend, backend,n_qubit,n_shot):
    print("params in def chemistry in vqe.py: ", params)
    # assert(len(params)%2==0)
    width_len = int(len(params)-1*(n_qubit-1))
    split_ind = int(width_len/2)
    amp = np.array(params[:split_ind])
    angle = np.array(params[split_ind:width_len])*np.pi*2
    width_1 = (np.array(params[width_len:]))
    num_items = (1024 - 256) // 16 + 1
    width_norm = (width_1 - 256) / (1024 - 256)
    width_norm = np.clip(width_norm, 0, 1)
    width = (np.round(width_norm * (num_items - 1)) * 16 + 256).astype(int)
    amp = amp.tolist()
    angle = angle.tolist()
    width = width.tolist()
    keys = [key for key in pauli_dict]
    values = [pauli_dict[key] for key in pauli_dict]
    expect_values = []

    for key, value in zip(keys, values):
        prepulse = HE_pulse(backend, amp, angle, width)
        expect = vqe_one(prepulse, n_qubit, n_shot, pulse_backend, backend, key, value)
        expect_values.append(expect)
    print("E for cur_iter: ",sum(expect_values))
    return sum(expect_values)