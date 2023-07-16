from qiskit.utils import QuantumInstance
from qiskit import Aer


qasm_sim  = Aer.get_backend('qasm_simulator')
sv_sim    = Aer.get_backend('statevector_simulator')
uni_sim   = Aer.get_backend('unitary_simulator')

def make_qi(simulator, shots=8192):
    return QuantumInstance(backend=simulator, shots=shots)

qasm_qi = make_qi(qasm_sim)
sv_qi   = make_qi(sv_sim)
uni_qi  = make_qi(uni_sim)

