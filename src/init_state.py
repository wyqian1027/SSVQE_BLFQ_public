from qiskit import QuantumCircuit

def build_ref_state_for_UCC(occ_locs, n_qubit):
    ''' Build the initial state for UCC ansatz. '''
    circ = QuantumCircuit(n_qubit)
    for occ_loc in occ_locs:
        circ.x(occ_loc)
    return circ

def make_quantum_state_str(n_qubit, state_idx):
    ''' Return a bitstring to represent a quantum circuit. '''
    assert 0 <= state_idx < 2**n_qubit
    b = bin(state_idx)[2:]
    b = "0"*(n_qubit-len(b)) + b
    return b

def make_quantum_state(n_qubit, state_idx):
    ''' Make a single QuantumCircuit object for SSVQE initial reference state. '''
    b = make_quantum_state_str(n_qubit, state_idx)
    b_idx = [n_qubit-1-i for i, c in enumerate(b) if c == "1"]
    qs = QuantumCircuit(n_qubit)
    if b_idx: qs.x(b_idx)
    return qs

def get_ref_states(n_qubit, n_state, cutoff=8):
    ''' Make a list of QuantumCircuit object for SSVQE initial reference states.'''
    assert 1 <= n_state <= min(cutoff, 2**n_qubit)
    return [make_quantum_state(n_qubit, state_idx) for state_idx in range(0, n_state)]