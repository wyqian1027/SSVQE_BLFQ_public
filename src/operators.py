from functools import lru_cache
import numpy as np
import re
from math import log, sqrt, pi

from qiskit.opflow import Z, I, X, Y
from qiskit.opflow import StateFn
from qiskit.opflow import CircuitSampler, PauliExpectation
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.quantum_info import Pauli



class MatrixEncoding:

    def __init__(self, matrix, encoding='compact'):
        self.matrix = matrix
        self.encoding = encoding.lower()
        self.operator = self.encode()

    def encode(self):
        if self.encoding == 'compact':
            self.operator = convert_matrix_to_operator(self.matrix)
        else:
            self.operator = qubitize_H(self.matrix)
        return self.operator
    
    def get_operator(self):
        return self.operator




# ================ direct encoding ================

def qubitize_H(ham):
    ''' Return the Hamiltonian on qubit space 
        - ham: either a list of list or 2D np.array
    '''
    assert len(ham[0]) == len(ham)
    n_qubit = len(ham)
    res = 0
    for i in range(n_qubit):
        for j in range(n_qubit):
            op = (operator_ca_pair(i, j, n_qubit) * ham[i][j]).reduce()
            res += op
    return res.reduce()

@lru_cache(maxsize=None)
def operator(loc, n_qubit, creation=True):
    site = (X - 1j*Y) / 2 if creation else (X + 1j*Y) / 2
    op = None
    for i in range(n_qubit):
        if i < loc:
            if op == None:
                op = Z
            else:
                op = op ^ Z
        elif i == loc:
            if op == None:
                op = site
            else:
                op = op ^ site
        else:
            op = op ^ I
    return op

@lru_cache(maxsize=None)
def operator_ca_pair(c_loc, a_loc, n_qubit):
    c = operator(c_loc, n_qubit)
    a = operator(a_loc, n_qubit, False)
    return (c @ a).reduce()

@lru_cache(maxsize=None)
def operator_ucc_T1_pair(c_loc, a_loc, n_qubit):
    return (operator_ca_pair(c_loc, a_loc, n_qubit) - operator_ca_pair(a_loc, c_loc, n_qubit)).reduce()

# ================ compact encoding ================

@lru_cache(maxsize=None)
def make_pauli_str(size):
    pauli_strs = ['I', 'X', 'Y', 'Z']
    if size == 1: return pauli_strs
    return [pre + p for pre in make_pauli_str(size-1) for p in pauli_strs]      

def get_pauliOp_from_matrix_by_basis_str(basis_str, matrix):
    n = len(matrix)
    pauli = Pauli(basis_str)
    prod = matrix.dot(pauli.to_matrix())
    trace_value = sum([prod[i][i] for i in range(n)])/n
    trace_value = trace_value  # SHOULD NEVER ROUND!
    return PauliOp(pauli, coeff=trace_value)

def convert_matrix_to_operator(matrix, tol=1e-8):
    n = len(matrix); pauli_n = int(log(n, 2))
    assert n == 2**pauli_n
    out = 0
    for basis_str in make_pauli_str(pauli_n):
        pauliOp = get_pauliOp_from_matrix_by_basis_str(basis_str, matrix)
        if np.abs(pauliOp.coeff) <= tol: continue
        out += pauliOp
    return out


# decay constant / distribution function encoding
def get_inner_v_matrix(pos_idx, neg_idx, size):
    basis_v = np.zeros(size)
    assert set(pos_idx) & set(neg_idx) == set([])
    for idx in pos_idx:
        basis_v[idx] = 1
    for idx in neg_idx:
        basis_v[idx] = -1
    basis_v_col = basis_v.reshape(size,-1)
    basis_v_row = basis_v.reshape(1,size)
    inner_v = basis_v_col.dot(basis_v_row)
    return inner_v

def get_inner_v_qubit(pos_idx, neg_idx, size):
    return qubitize_H(get_inner_v_matrix(pos_idx, neg_idx, size))

# LFWF amplitudes
def get_amp_operator(amp_idx=0, n_qubits=4):
    m = np.zeros((n_qubits,n_qubits))
    m[amp_idx][amp_idx] = 1
    return qubitize_H(m)
