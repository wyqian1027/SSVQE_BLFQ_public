import numpy as np
from functools import lru_cache
from itertools import combinations
from matplotlib import pyplot as plt

from qiskit import QuantumCircuit
from qiskit.opflow import StateFn, CircuitSampler
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_city, plot_bloch_multivector, plot_state_hinton
from qiskit.opflow.state_fns import SparseVectorStateFn

from blfq import get_decay_coeff
from operators import convert_matrix_to_operator
from solver_SSVQE import get_exp_val
# eigenstates:

def get_eigenstate_from_params(ansatz, params, state, sampler=None):
    assert sampler == None or type(sampler) == CircuitSampler
    evolved_state = StateFn(state.compose(ansatz.bind_parameters(params)))
    if sampler == None:
        eval_state = evolved_state.eval()
    else:
        eval_state = sampler.convert(evolved_state).eval()
    if type(eval_state) == SparseVectorStateFn:
        # probability level now...
        return Statevector(eval_state.to_matrix().flatten())
    else:
        # amplitude level density matrix
        return eval_state.primitive
    # return eval_state

def get_eigenstate_from_params_2(ansatz, params, state, sampler=None):
    assert sampler == None or type(sampler) == CircuitSampler
    evolved_state = StateFn(state.compose(ansatz.bind_parameters(params)))
    if sampler == None:
        eval_state = evolved_state.eval()
    else:
        eval_state = sampler.convert(evolved_state).eval()
    return eval_state


class BLFQ_Quantum:
    def __init__(self, ansatz, params, ref_states, sampler=None):
        assert sampler == None or type(sampler) == CircuitSampler
        assert len(ref_states) >= 1 and type(ref_states[0]) == QuantumCircuit
        self.ansatz = ansatz
        self.params = params
        self.ref_states = ref_states
        self.sampler = sampler
        self._sampler_for_eigenvector = None
        self.eigenstate_dict = {}
        self.n = len(self.ref_states)
        self.statelist = [f'state_{i}' for i in range(self.n)]

    def get_eigenstate(self, idx=0):
        if idx not in self.eigenstate_dict:
            self.eigenstate_dict[idx] = get_eigenstate_from_params(self.ansatz, self.params, self.ref_states[idx], 
            self._sampler_for_eigenvector)
        return self.eigenstate_dict[idx]
    
    def draw_eigenstate(self, idx=0, output='print'):
        state_vec = get_eigenstate_from_params(self.ansatz, self.params, self.ref_states[idx], 
        self._sampler_for_eigenvector)
        if output == 'print':
            return print_state(state_vec)
        elif output == 'bloch':
            return plot_bloch_multivector(state_vec)
        elif output == 'hinton':
            return plot_state_hinton(state_vec)

    def show_orthogonality(self, round_level=6):
        all_svs = [self.get_eigenstate(idx=i) for i in range(self.n)]
        total = 0
        for i in range(len(all_svs)-1):
            for j in range(i+1, len(all_svs)):
                product = np.dot(np.conj(all_svs[i].data), all_svs[j].data)
                # print(i, j, product)
                total += np.abs(product)
                print(rf'< psi_{i} | psi_{j} > = {np.round(product, round_level)}')
        return total / (len(all_svs)) / (len(all_svs)-1) * 2

    def show_normality(self, round_level=6):
        all_svs = [self.get_eigenstate(idx=i) for i in range(self.n)]
        total = 0
        for i in range(len(all_svs)):
            product = np.dot(np.conj(all_svs[i].data), all_svs[i].data)
            total += np.abs(product)
            print(rf'< psi_{i} | psi_{i} > = {np.round(product, round_level)}')
        return total / (len(all_svs))

    def show_corr(self, verbose=0):
        all_svs = [self.get_eigenstate(idx=i) for i in range(self.n)]
        matrix = np.zeros((len(all_svs), len(all_svs)))
        for i in range(len(all_svs)):
            for j in range(len(all_svs)):
                if verbose == 1: print(i, j, np.dot(np.conj(all_svs[i].data), all_svs[j].data))
                matrix[i][j] = np.abs(np.dot(np.conj(all_svs[i].data), all_svs[j].data))
        # plt.matshow(matrix)
        # plt.show()
        figure = plt.figure()
        axes = figure.add_subplot(111)
        
        # using the matshow() function
        caxes = axes.matshow(matrix, interpolation ='nearest')
        figure.colorbar(caxes)
        
        axes.set_xticklabels(['']+self.statelist)
        axes.set_yticklabels(['']+self.statelist)
        
        plt.show()



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


def construct_decay_op_from_blfq(blfq_file, idx, decay='p', verbose=0):
    mf, kap = blfq_file.mf, blfq_file.kap
    evect = blfq_file.get_evect(idx)
    assert decay in ['p', 'v']
    decay_vector = []
    for row in evect:
        n,m,l,s1,s2 = row[:5].astype(int); coeff = row[-1]
        # l % 2 == 0 because of parity constraint
        if m == 0 and s1 == 1 and s2 == -1 and l % 2 == 0:
            val = get_decay_coeff(mf,kap,Nc=3,n=n,l=l)
        elif m == 0 and s1 == -1 and s2 == 1 and l % 2 == 0:
            val = get_decay_coeff(mf,kap,Nc=3,n=n,l=l)
            val = -val if decay=='p' else val
        else:
            val = 0
        decay_vector.append(val)
    decay_vector = np.array(decay_vector)
    assert len(decay_vector) == len(evect)
    size = len(evect)
    col_v = decay_vector.reshape(size,-1)
    row_v = decay_vector.reshape(1,size)
    inner_v = col_v.dot(row_v)
    if verbose: print(inner_v)
    return convert_matrix_to_operator(inner_v)
            
            
def get_decay_from_sim(ansatz, params, decay_op, state, sampler=None, exp_method='factory'):
    ''' Get decay constant in MeV from circuit'''
    return np.sqrt(get_exp_val(ansatz, params, decay_op, state, sampler=sampler, exp_method=exp_method))*1000
    

from collections import defaultdict
from blfq import chi
def construct_pdf_op_from_blfq(blfq_file, idx, x, verbose=0):
    mf, kap = blfq_file.mf, blfq_file.kap
    mu = 4*mf**2/kap**2
    evect = blfq_file.get_evect(idx)
    size = len(evect)
    groups = defaultdict(list)
    for row_id, row in enumerate(evect):
        n,m,l,s1,s2 = row[:5].astype(int); coeff = row[-1]
        groups[(n,m,s1,s2)].append([n, m, l, s1, s2, coeff, row_id])
    # all_ns = sorted(set(list(evect[:,0].astype(int))))
    # all_ms = sorted(set(list(evect[:,1].astype(int))))
    # all_ls = sorted(set(list(evect[:,2].astype(int))))
        
    pdf_mat = np.zeros((size, size))
    res = 0
    for group_key, group in groups.items():
        if verbose==1: print("Group: (n,m,s1,s2) =", group_key)
        for i in range(len(group)):
            for j in range(len(group)):
                l1, l2 = int(group[i][2]), int(group[j][2])
                c1, c2 = group[i][5], group[j][5]
                idx1, idx2 = group[i][6], group[j][6]
                pdf_mat[idx1,idx2] += chi(l1, mu, x)*chi(l2, mu, x)
    if verbose:
        print(np.round(pdf_mat,3))
                            
    return convert_matrix_to_operator(pdf_mat)                
            
def get_pdf_from_sim(ansatz, params, pdf_op, state, sampler=None, exp_method='factory'):
    ''' Get PDF from PDF operator from circuit'''
    return get_exp_val(ansatz, params, pdf_op, state, sampler=sampler, exp_method=exp_method)
    