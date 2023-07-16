from math import pi, log
import numpy as np
from functools import lru_cache
import pickle
from datetime import datetime
from time import time, sleep
from itertools import combinations
import re, os

from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit


# Simple File IO
class FileIO:
    def __init__(self, filename):
        self.filename = filename
        self.exists()
    
    def exists(self):
        if not os.path.exists(self.filename):
            open(self.filename, 'a').close()
        return True
    
    def clear(self):
        return open(self.filename, "w").close()

    def append_text(self, text):
        with open(self.filename, 'a') as f:
            f.write(text +"\n")

    def head(self, first_n):
        with open(self.filename, 'r') as f:
            data = f.readlines()[:first_n]
        for line in data:
            print(line)

# load spectroscopy/eigenvalue file
def load_Ham_data(f_path):
    output = []
    with open(f_path, "r") as f:
        output = f.readlines()
    return np.array([list(map(float, re.split('\s+', line.strip()))) for line in output])

def load_Ham_data_by_threshold(f_path, abs_threshold=1e-8):
    output = []
    with open(f_path, "r") as f:
        output = f.readlines()
    Ham = np.array([list(map(float, re.split('\s+', line.strip()))) for line in output])
    mask = np.abs(Ham) <= abs_threshold
    Ham[mask] = 0
    return Ham

def get_eigenstate_by_threshold(eigenstate, round=True, abs_threshold=1e-8):
    mask = np.abs(eigenstate) <= abs_threshold
    new_eig = np.copy(eigenstate)
    new_eig[mask] = 0
    if round: new_eig = np.around(new_eig, 3)
    return new_eig

# write text
def write_text_to_file(text_str, filename):
    filename = f"{filename}_{get_time_stamp()}.txt"
    with open(filename, 'w') as f:
        f.writelines(text_str)
    return True

@lru_cache(maxsize=None)
def get_list_of_qubit_states(n_qubit, braket=False):
    ''' return a list of 2**n_qubit qubit state, given n_qubit '''
    if n_qubit <= 0: 
        return []
    elif n_qubit == 1:
        res = ["0", "1"]
    else:
        res = [''.join([l, r]) 
                for l in ['0', '1'] 
                for r in get_list_of_qubit_states(n_qubit-1)]
    if braket:
        return [f'|{r}>' for r in res]
    return res
            

def print_state(statevector, display=False, round_level=3, atol=1e-8):
    assert type(statevector) == Statevector
    N = statevector.dim
    n_qubit = statevector.num_qubits #int(log(N, 2))
    assert N == 2**n_qubit
    assert statevector.is_valid() # normalization check
    basis = get_list_of_qubit_states(n_qubit, braket=True)
    res = ""
    for i in range(N):
        amp = statevector.data[i]
        if np.abs(amp) < atol: 
            amp = 0
        else:
            amp = np.round(amp, round_level)
        component = f"{amp} * {basis[i]}"
        if display: print("state {:>3}: ".format(i), component)
        if amp != 0:
            if res != "": res += " + "
            res = res + f"{component}" 
    return res

def get_state_amplitude_list(statevector, real=True):
    assert type(statevector) == Statevector
    return np.real(statevector.data) if real else statevector.data


def is_orthogonal(statevector1, statevector2, atol=1e-8):
    return np.isclose(np.abs(inner_prod_states(statevector1, statevector2)), 0, atol)

def inner_prod_states(statevector1, statevector2):
    assert type(statevector1) == Statevector and type(statevector2) == Statevector
    assert statevector1.dim == statevector2.dim
    return np.dot(statevector1.data, statevector2.data)


def get_all_inner_prod_among_states(statevector_list):
    return [np.dot(sv1.data, sv2.data) 
            for sv1, sv2 in combinations(statevector_list, 2)]


def get_ref_state_by_bit_str(bit_str, num_qubits=4):
    ''' note: bit_str order = q0, q1, q2, ...'''
    state = QuantumCircuit(num_qubits)
    for j, c in enumerate(bit_str):
        if c == '1': state.x(j)
    return state
    
def get_bit_str(x, size, reverse=False):
    bit_str = bin(x)[2:]
    bit_str = (size - len(bit_str))*'0' + bit_str
    if reverse: bit_str = bit_str[::-1]
    return bit_str

def get_orth_ref_states(n_state=1, num_qubits=4):
    ''' note: bit_str order = | q0 q1 q2 ... >'''
    states = []
    for i in range(n_state):
        bit_str = get_bit_str(i+1, num_qubits, reverse=True)
        print("{:>2}: {:>5}".format(i, bit_str))
        states.append(get_ref_state_by_bit_str(bit_str, num_qubits))
    return states


def save_to_pickle(data, filename=None):
    if filename == None or filename == "": 
        filename = "Results_{}.pickle".format(get_time_stamp())
    else:
        filename = "{}_{}.pickle".format(filename, get_time_stamp())
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved data to {filename}.")
    return filename    

def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(f"Loaded data from {filename}.")
    return obj  

def get_time_stamp():
    # return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return datetime.now().strftime("%m_%d_%H_%M_%S")

def format_time(t):
    s = t % 60
    m = (t // 60) % 60
    h = t //3600
    return f"{h:>2.0f}h {m:>2.0f}m {s:>2.0f}s"

def time_since(t):
    return format_time(time() - t)

def job_sleeper(t=10):
    sleep(t)


# def get_version():
#     import qiskit.tools.jupyter
#     # %qiskit_version_table
#     # %qiskit_copyright