import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.compiler import transpile


class UCCAnsatz:
    def __init__(self, n_qubit, occ_locs, vir_locs, decompose=True, barrier=True):
        '''
        Unitary Coupled Cluster Ansatz implementation.
        - occ_locs: occupied locations, must be a list
        - vir_locs: virtual locations, must be a list
        '''
        self.circ = build_ucc_circ(occ_locs, vir_locs, n_qubit, decompose, barrier=barrier)
    
    def summary(self):
        print(f"Total gates: {len(circ.data)}; Total depth: {circ.depth()}; Gates: " , circ.count_ops())

    def get_circ(self):
        return self.circ


class HardwareEffAnsatz:
    def __init__(self, n_qubit, depth1, depth2, barrier=False, input_states='all'):
        '''
        Hardware Efficient Ansatz implementation with CZ two-qubit gates and RX, RZ single-qubit gates.
        '''
        self.input_states = list(range(n_qubit)) if input_states == 'all' else input_states
        self.circ = he_ansatz(n_qubit, depth1, depth2, self.input_states, barrier)
    
    def summary(self):
        print(f"Total gates: {len(circ.data)}; Total depth: {circ.depth()}; Gates: " , circ.count_ops())

    def get_circ(self):
        return self.circ


def HEA_ansatz(n, l, pauli_list, final_rotation=True, barrier=False):
    num_p = n*l*len(pauli_list) if final_rotation == False else n*(l+1)*len(pauli_list)
    p = ParameterVector('p', length=num_p)
    qc = QuantumCircuit(n)
    def add_rotation_layer(qc, pauli_list, p_idx):
        c = p_idx
        for i in range(n):
            for pauli in pauli_list:
                if pauli == 'rx':
                    qc.rx(p[c], i); c += 1
                elif pauli == 'ry':
                    qc.ry(p[c], i); c += 1
                elif pauli == 'rz':
                    qc.rz(p[c], i); c += 1
    def add_entanglement(qc):
        for i in range(1, n, 2):
            qc.cx(i-1, i)
        for i in range(2, n, 2):
            qc.cx(i-1, i)
    
    p_idx = 0
    for _ in range(l):
        add_rotation_layer(qc, pauli_list, p_idx)
        p_idx += n*len(pauli_list)
        add_entanglement(qc)
        if barrier: qc.barrier()
    if final_rotation:
        add_rotation_layer(qc, pauli_list, p_idx)
        p_idx += n*len(pauli_list)
    assert p_idx == num_p    
    return qc

def ALT_ansatz(n, m, l, pauli_list, barrier=False, block_depth=2, random_pauli=False, final_rotation=False):
    '''https://arxiv.org/pdf/2005.12537.pdf'''
    num_p = n*l*block_depth*len(pauli_list) if final_rotation==False else n*l*block_depth*len(pauli_list) + n
    # print(f"Total of {num_p} parameters")
    p = ParameterVector('p', length=num_p)
    qc = QuantumCircuit(n, name=f"ALT_{n}_{m}_{l}")
    def add_block_over_qubits(qc, pauli_list, qubit_idxs, block_depth, p_idx, entangle=True):
        c = p_idx
        for _ in range(block_depth):
            for qubit_i in qubit_idxs:
                # rotations
                for pauli in pauli_list:
                    if random_pauli == False:
                        if pauli == 'rx':
                            qc.rx(p[c], qubit_i); c += 1
                        elif pauli == 'ry':
                            qc.ry(p[c], qubit_i); c += 1
                        elif pauli == 'rz':
                            qc.rz(p[c], qubit_i); c += 1
                    else:
                        u = np.random.random()
                        if u <= 1/3:
                            qc.rx(p[c], qubit_i); c += 1
                        elif u <= 2/3:
                            qc.ry(p[c], qubit_i); c += 1
                        else:
                            qc.rz(p[c], qubit_i); c += 1
                # entanglement
            if entangle == False: return
            for i, qubit_i in enumerate(qubit_idxs):
                if i == 0: continue
                qc.cx(qubit_idxs[i-1], qubit_i) # need to change when n is big
    
    p_idx = 0
    for l_idx in range(l):
        # alternating layers
        offset = 0 if (l_idx % 2) == 0 else m // 2
        qubit_lists = [[block_i*m + j - offset for j in range(m)] for block_i in range(n//m)]
        if (l_idx % 2) == 1:
            qubit_lists[0] = list(range(m//2))
            qubit_lists.append(list(range(n-m//2, n)))
        # print(qubit_lists)    
        
        for qubit_idxs in qubit_lists:
            add_block_over_qubits(qc, pauli_list, qubit_idxs, block_depth, p_idx)
            p_idx += len(qubit_idxs)*len(pauli_list)*block_depth
        if barrier: qc.barrier()
    
    if final_rotation == True:
        add_block_over_qubits(qc, ['ry'], list(range(n)), 1, p_idx, entangle=False)
        p_idx += n

    assert p_idx == num_p    
    return qc


    
# ================== Aux functions =====================


# ucc ansatz
def build_ucc_subcirc(occ_loc, vir_loc, n_qubit, params, param_index, decompose=True):
    '''
    build ucc circuit for a^\dagger_v a_o - a^\dagger_o a_v 
    * occ_loc = a_i     = a_loc
    * vir_loc = a_dag_j = c_loc
    '''
    assert occ_loc < n_qubit and vir_loc < n_qubit and vir_loc != occ_loc
    circ = QuantumCircuit(n_qubit)
    # if you dont care about param meaning, the sign is not needed
    angle = params[param_index] if vir_loc > occ_loc else -params[param_index]
    # without loss of generality, since U_{1->0} = -U(0->1)
    max_loc = max(occ_loc, vir_loc)
    min_loc = min(occ_loc, vir_loc)
    vir_loc = max_loc
    occ_loc = min_loc
    
    def entanglement_block(circ, occ_loc, vir_loc, n_qubit, angle):
        for i in range(occ_loc, vir_loc):
            circ.cx(i, i+1)
        circ.rz(angle, vir_loc)
        for i in range(vir_loc-1, occ_loc-1, -1):
            circ.cx(i, i+1) 
    
    def rotation_block(circ, occ_loc, vir_loc, n_qubit, first_part=True, conjugate=False):
        if first_part: 
            # apply Rx(-90) on Y (or occ), apply H on X (or vir); conjugate unitary if reverse operation
            circ.rx(-np.pi/2 if conjugate==False else np.pi/2, occ_loc)
            circ.h(vir_loc)
        else:
            # apply Rx(-90) on Y (or vir), apply H on X (or occ); conjugate unitary if reverse operation
            circ.rx(-np.pi/2 if conjugate==False else np.pi/2, vir_loc)
            circ.h(occ_loc)
    
    # first part
    rotation_block(circ, occ_loc, vir_loc, n_qubit, first_part=True, conjugate=False)
    entanglement_block(circ, occ_loc, vir_loc, n_qubit, angle)
    rotation_block(circ, occ_loc, vir_loc, n_qubit, first_part=True, conjugate=True)
    
    # second part
    rotation_block(circ, occ_loc, vir_loc, n_qubit, first_part=False, conjugate=False)
    entanglement_block(circ, occ_loc, vir_loc, n_qubit, -angle)
    rotation_block(circ, occ_loc, vir_loc, n_qubit, first_part=False, conjugate=True)
    if decompose == True: circ = circ.decompose()
    return circ

def build_ucc_circ(occ_locs, vir_locs, n_qubit, decompose=True, barrier=True):
    if type(occ_locs) != list: occ_locs = [occ_locs]
    if type(vir_locs) != list: vir_locs = [vir_locs]
    assert set(occ_locs) & set(vir_locs) == set()
    params = ParameterVector('θ', len(occ_locs)*len(vir_locs))
    param_index = 0
    
    circ = QuantumCircuit(n_qubit)
    for occ_loc in occ_locs:
        for vir_loc in vir_locs:
            circ = circ.compose(build_ucc_subcirc(occ_loc, vir_loc, n_qubit, params, param_index))
            if barrier: circ.barrier()
            param_index += 1
    
    if decompose == True: circ = circ.decompose()
    return circ


def build_optim_ucc_ansatz(occ_locs, vir_locs, n_qubit, optim_level=3, basis_gates=['u1', 'u2', 'u3', 'cx']):
    ansatz = build_ucc_circ(occ_locs, vir_locs, n_qubit)
    ansatz = get_transpiled_circ(ansatz, optim_level=optim_level, basis_gates=basis_gates)
    return ansatz
        
def get_transpiled_circ(circ, optim_level=3, basis_gates=['u1', 'u2', 'u3', 'cx']):
    transpiled_circ = transpile(circ, basis_gates=basis_gates, optimization_level=optim_level)
    return transpiled_circ

# he ansatz
def he_ansatz(n_qubit, depth1, depth2, input_states, is_barrier=False):
    input_states = list(sorted(input_states))
    n_phi = len(input_states)*2*depth1
    n_theta = n_qubit*2*(depth2+1)
    phi = ParameterVector('a', n_phi)
    theta = ParameterVector('b', n_theta)
    circuit = QuantumCircuit(n_qubit)
    # first stack (phi)
    idx = 0
    for d in range(depth1):
        for input_state in input_states:
            circuit.rx(phi[idx],   input_state)
            circuit.rz(phi[idx+1], input_state)
            idx += 2
        if len(input_states) >= 2:
            for i in range(len(input_states)-1):
                circuit.cz(input_states[i], input_states[i+1])
    if is_barrier: circuit.barrier()
    assert idx == n_phi
    # second stack (theta)
    idx = 0
    for d in range(depth2):
        for i in range(n_qubit):
            circuit.rx(theta[idx],   i)
            circuit.rz(theta[idx+1], i)
            idx += 2
        for i in range(n_qubit-1):
            circuit.cz(i, i+1)
    if is_barrier: circuit.barrier()

    for i in range(n_qubit):
        circuit.rx(theta[idx],   i)
        circuit.rz(theta[idx+1], i)
        idx += 2
    assert idx == n_theta
    return circuit

