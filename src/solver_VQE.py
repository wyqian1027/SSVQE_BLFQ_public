import numpy as np
from time import time

# qiskit
from qiskit.algorithms import VQE as qiskitVQE
from qiskit.circuit import QuantumCircuit
from qiskit.utils import algorithm_globals
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, SPSA, QNSPSA

# local
from simulator import qasm_qi, sv_qi, qasm_sim, make_qi
from optimization_result import OptimizationResult
from tools import get_time_stamp, FileIO, time_since


class VQE:

    def __init__(self, ansatz, operator, ref_state=None, optimizer_name="COBYLA",
                quantum_instance=None, initial_point=None, 
                random_seed=None, verbose=0, observable_ops=[], max_iter=500):
        ''' 
        VQE program based on qiskit.algorithms.VQE
        - sampling is based on quantum instance, shots are 8192 by default 
        - observable_ops can be supplied 
        - ref_state can be specified apart from ansatz
        '''  
        self.ansatz = ansatz
        self.operator = operator
        self.n_qubits = operator.num_qubits
        self.ref_state = ref_state
        self.max_iter = max_iter
        self.optimizer_name = optimizer_name.upper()
        self.optimizer = self.set_optimizer()
        self.qi = quantum_instance if quantum_instance != None else make_qi(qasm_sim, 20000)
        self.random_seed = random_seed
        if self.random_seed != None: 
            algorithm_globals.random_seed = random_seed
        self.observable_ops = observable_ops
        self.verbose = verbose
        self.init_pts = initial_point
        self.text_log = ""
        self.full_ansatz = ansatz if ref_state == None else ref_state.compose(ansatz)
        
        self.vqe = qiskitVQE(self.full_ansatz, self.optimizer, quantum_instance=self.qi, 
                             initial_point=self.init_pts, callback=self.callback)
        
        self.optim_result = OptimizationResult(optimizer_name=self.optimizer_name)
        self.time_stamp = get_time_stamp()
        self.fileHander = FileIO(f'Job_{self.__str__()}_ID_{self.time_stamp}.log')
 
    def __str__(self):
        return f'vqe_{self.n_qubits}qubit_{self.optimizer_name}'
    def get_time_stamp(self):
        return self.time_stamp

    def set_optimizer(self):
        ''' Customize available optimizers'''
        if self.optimizer_name == 'COBYLA':
            return COBYLA(maxiter=self.max_iter)
        elif self.optimizer_name == "L_BFGS_B":
            return L_BFGS_B(maxiter=self.max_iter, maxfun=50*self.max_iter, ftol=1e-9,
                            options={"maxls": 20, 'disp': False})
        elif self.optimizer_name == "SLSQP":
            return SLSQP(maxiter=self.max_iter)
        elif self.optimizer_name == 'SPSA':
            return SPSA(maxiter=self.max_iter)#, termination_checker=self.terminate_func)
        elif self.optimizer_name == 'QNSPSA':
            return QNSPSA(maxiter=self.max_iter, fidelity=QNSPSA.get_fidelity(self.ansatz))

    def run(self):
        self.eval_hist, self.param_hist, self.error_hist, self.cost_hist = [], [], [], []
        self.start_time = time()
        self.vqe_result = self.vqe.compute_minimum_eigenvalue(operator=self.operator, aux_operators=self.observable_ops)
        self.final_param = self.vqe_result.optimal_point
        self.final_eval = self.vqe_result.optimal_value
        self.final_error = self.error_hist[-1][0]
        self.optim_result.add_result(self.param_hist, self.eval_hist, self.cost_hist, self.error_hist, 
                                     self.final_param, self.final_eval,
                                     self.vqe_result.optimizer_time, self.vqe_result)
        if self.verbose >= 1:
            print('\nParameters: \n', self.final_param, '\n')
        if self.verbose >= 0:
            print('Optimization completed!')
            print("| {:4} iters | {:.2f}s | evals = {:.3f} ({:.3f}) |".format(
                self.optim_result.n_iter[-1], self.optim_result.time[-1], np.real(self.final_eval), np.real(self.final_error)))
        return self.optim_result
        
    def callback(self, eval_count, params, mean, error):
        self.eval_hist.append([mean])
        self.cost_hist.append([mean])  # For VQE, cost = eval
        self.param_hist.append(params)
        self.error_hist.append([error])
        # if self.verbose >= 2:
        #     print("|{:>03}| cost = {:.3f}".format(len(self.param_hist), mean))
        text = "|{:>4}| {} | cost = {:.3f} ({:.3f}) | evals = {:.3f}".format(
            len(self.param_hist), time_since(self.start_time), mean, 
            error, mean)
        self.text_log += text + "\n"
        self.fileHander.append_text(text)
        if self.verbose >= 1:          
            print(text)

    # def callback_scipy(self, param):
    #     ''' Callback function for Scipy optimizer'''
    #     # assert np.all(np.array(param) == self.current_param)
    #     self.eval_hist.append(self.current_eval)
    #     self.param_hist.append(self.current_param)
    #     self.cost_hist.append([self.current_cost])
    #     self.error_hist.append(self.current_error)
    #     self.n_iter += 1
    #     text = "|{:>4}| {} | cost = {:.3f} ({:.3f}) | evals = {}".format(
    #         len(self.param_hist), time_since(self.start_time), self.current_cost, 
    #         self.current_error, list(np.round(self.current_cost, 0).astype(int)))
    #     self.text_log += text + "\n"
    #     self.fileHander.append_text(text)
    #     if self.verbose >= 1:          
    #         print(text)

    # def callback_qiskit(self, num_eval, param, f_eval, step, acc):
    #     ''' Callback function for SPSA, QNSPSA, etc'''
    #     return self.callback_scipy(param)

