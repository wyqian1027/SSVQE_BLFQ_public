from time import time
import os
from datetime import datetime
import numpy as np


from qiskit import transpile, execute
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import Z, I, X, Y, StateFn, CircuitStateFn
from qiskit.opflow import CircuitSampler, PauliExpectation, ExpectationFactory
from qiskit.utils import QuantumInstance, algorithm_globals

# optimizers
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, SPSA, QNSPSA

# solvers
# from qiskit.algorithms import VQE
from optimization_result import OptimizationResult
from tools import write_text_to_file

from mitigation import MeasCalibration
from measurement import PostMeasurement, PauliMeasStrSum
from tools import FileIO, time_since, job_sleeper, get_time_stamp
from qiskit.tools import job_monitor
from qiskit.opflow import StateFn


# Generic SSVQE solver working with local simulator.

class SSVQE_solver:

    def __init__(self, ansatz, operator, ref_state=[None], optimizer_name="COBYLA", max_iter=500,
                quantum_instance=None, exp_method='factory', initial_point=None, 
                random_seed=None, verbose=0, weights='default'):
        ''' 
        Subspace-Search VQE (SSVQE) program based on reference: Nakanishi, Mitarai, Fujii (https://arxiv.org/abs/1810.09434)
        - ref_state should be a non-empty list of initial quantum circuits
        - weights can be supplied with a numpy array
        - for a proof of principle, one can set quantum instance = None, exact simulation will follow
        - for a realistic quantum sampling results, please set quantum instance = qasm, exp_method = 'pauli'
        '''  
        self.ansatz = ansatz
        self.full_ansatz = self.ansatz
        self.operator = operator
        self.n_qubits = operator.num_qubits
        self.ref_state = ref_state
        self.n_ref = len(self.ref_state)
        self.max_iter = max_iter
        self.optimizer_name = optimizer_name.upper()
        self.optimizer = self.set_optimizer()
        self.qi = quantum_instance 
        self.sampler = None if self.qi == None else CircuitSampler(quantum_instance)
        self.exp_method = exp_method
        self.random_seed = random_seed
        if self.random_seed != None: 
            algorithm_globals.random_seed = random_seed
        self.verbose = verbose
        self.init_pts = initial_point
        self.weights = self.set_weights() if weights == 'default' else weights
        self.validation()

        self.optim_result = OptimizationResult(optimizer_name=self.optimizer_name)
        self.time_stamp = get_time_stamp()
        self.fileHander = FileIO(f'Job_{self.__str__()}_ID_{self.time_stamp}.log')

    def get_time_stamp(self):
        return self.time_stamp

    def validation(self):
        assert type(self.ref_state) == list
        assert self.operator.num_qubits == self.ansatz.num_qubits
        for state in self.ref_state:
            assert state.num_qubits == self.operator.num_qubits
        return True

    def __str__(self):
        return f'ssvqe_{self.n_qubits}qubit_{len(self.ref_state)}ref_{self.optimizer_name}'

    def set_optimizer(self):
        ''' Customize available optimizers'''
        if self.optimizer_name == 'COBYLA':
            return COBYLA(maxiter=self.max_iter, callback=self.callback_scipy)
        elif self.optimizer_name == "L_BFGS_B":
            return L_BFGS_B(maxiter=self.max_iter, callback=self.callback_scipy, maxfun=50*self.max_iter, ftol=1e-9,
                            options={"maxls": 20, 'disp': False})
        elif self.optimizer_name == "SLSQP":
            return SLSQP(maxiter=self.max_iter, callback=self.callback_scipy)
        elif self.optimizer_name == 'SPSA':
            return SPSA(maxiter=self.max_iter, callback=self.callback_qiskit)#, termination_checker=self.terminate_func)
        elif self.optimizer_name == 'QNSPSA':
            return QNSPSA(maxiter=self.max_iter, fidelity=QNSPSA.get_fidelity(self.ansatz), callback=self.callback_qiskit)

    def set_weights(self):
        ''' Default weight factors '''
        return np.array([1/(2**i) for i in range(len(self.ref_state))])

    def set_initial_points(self, new_initial_points):
        ''' Reset initial points '''
        self.init_pts = new_initial_points
        return self.init_pts

    def run(self):
        ''' Run once '''
        if np.any(self.init_pts) == None: 
            print('Using random points\n')
            self.init_pts = self.set_initial_points(get_rand_init_pts(self.full_ansatz.num_parameters, self.random_seed))
        self.n_feval = self.n_iter = 0
        self.eval_hist, self.param_hist, self.error_hist, self.cost_hist = [], [], [], []
        self.text_log = ""
        t1 = time()
        self.vqe_result = self.minimize()
        t2 = time()
        self.final_param = self.vqe_result.x
        self.final_eval = self.current_eval
        self.final_error = self.current_error
        self.optim_result.add_result(self.param_hist, self.eval_hist, self.cost_hist, self.error_hist, 
                                     self.final_param, self.final_eval,
                                     t2- t1, self.vqe_result)
        summary_text = f'\nSummary of simulation ID = {len(self.optim_result)-1}\n'
        summary_text += '\nFinal Parameters: \n {}\n'.format(self.final_param)
        summary_text += f'\nOptimization completed!\n'
        summary_text += "| {:4} iters | {:.2f}s | {:4} fvals | {:.3f}s/fval | cost = {:.3f} |\n| evals = {} |".format(
            self.optim_result.n_iter[-1], self.optim_result.time[-1], self.n_feval, (t2-t1)/self.n_feval, self.current_cost,
            print_mean_and_error_together(self.final_eval, self.final_error))
        
        print(summary_text)
        self.fileHander.append_text(summary_text)

        return self.optim_result
        
    def callback_scipy(self, param):
        ''' Callback function for Scipy optimizer'''
        # assert np.all(np.array(param) == self.current_param)
        self.eval_hist.append(self.current_eval)
        self.param_hist.append(self.current_param)
        self.cost_hist.append([self.current_cost])
        self.error_hist.append(self.current_error)
        self.n_iter += 1
        text = "|{:>4}| {} | cost = {:.3f} ({:.3f}) | evals = {}".format(
            len(self.param_hist), time_since(self.start_time), self.current_cost, 
            np.dot(self.weights, self.current_error), list(np.round(self.current_eval, 0).astype(int)))
        self.text_log += text + "\n"
        self.fileHander.append_text(text)
        if self.verbose >= 1:          
            print(text)

    def callback_qiskit(self, num_eval, param, f_eval, step, acc):
        ''' Callback function for SPSA, QNSPSA, etc'''
        return self.callback_scipy(param)

    def terminate_func(self, num_eval, param, f_eval, step, acc):
        n_costs = len(self.cost_hist)
        if (n_costs >= 100) and (n_costs % 100) == 0:
            last_cost = self.cost_hist[-100][0]
            if f_eval <= last_cost*0.99:
                print(f'Still improving by 1 percent ({last_cost:.2f} => {f_eval:.2f}).')
                return False
            else:
                print(f"Terminated due to non-improvement ({last_cost:.2f} => {f_eval:.2f}).")
                return True
        return False

    def get_all_exp_vals(self, param):
        ''' Get expectation value for each state based on parameter, shape = (N, 2) '''
        return np.array([get_exp_val(self.full_ansatz, param, self.operator, state, self.sampler, self.exp_method)
                         for state in self.ref_state])

    def get_cost(self, param):
        ''' Get SSVQE cost, which is to be minimized (function evaluation)'''
        # print("already done with ", param)
        mean_and_err = self.get_all_exp_vals(param)
        self.current_eval = mean_and_err[:,0]
        self.current_error = mean_and_err[:,1]
        self.current_param = np.array(param)
        self.current_cost = np.dot(self.weights, self.current_eval)
        self.n_feval += 1
        if self.verbose >= 2: print(f"#feval: {self.n_feval:4}", self.current_eval, self.weights, self.current_cost) # debug purpose
        return self.current_cost

    def minimize(self):
        ''' Minimizer '''
        print('\nOptimization started...')
        print(self.init_pts)
        print()
        self.start_time = time()
        return self.optimizer.minimize(fun=self.get_cost, x0=self.init_pts)


def print_mean_and_error_together(means, errors):
    ''' Pretty print mean + uncertainty '''
    assert len(means) == len(errors)
    n = len(means)
    out = ""
    for i in range(n):
        out += "{}{:.2f} ({:.2f})".format(", " if i!=0 else "", means[i], errors[i])
    return out

def get_exp_val(ansatz, params, operator, state, sampler=None, exp_method='factory'):
    ''' Compute expectation value (with std) for a given operator, ansatz, and state. '''
    if sampler == None:
        evolved_ansatz = ansatz.bind_parameters(values=params)
        new_state = state.compose(evolved_ansatz)
        exp_val = StateFn(operator, is_measurement=True) @ StateFn(new_state)
        # exp_val = exp_val.reduce()
        mean, std = np.real(exp_val.eval()), 0
    else:
        if exp_method == 'factory':
            expectation = ExpectationFactory.build(
                        operator=operator,
                        backend=sampler.quantum_instance.backend)
            # if qasm + factory => AerPauliExpectation()
        elif exp_method=="pauli":
            expectation = PauliExpectation()

        evolved_ansatz = ansatz.bind_parameters(values=params)
        new_state = state.compose(evolved_ansatz)
        meas = expectation.convert(StateFn(operator, is_measurement=True))
        ansatz_circuit_op = CircuitStateFn(new_state)
        expect_op = meas.compose(ansatz_circuit_op).reduce()
        # using sampler to get mean +/- error
        
        sampled_expect_op = sampler.convert(expect_op)
        
        mean = np.real(sampled_expect_op.eval())
        # seem that if use pauli + statevector backend cannot compute variance
        # so ignoring the error for that case
        if exp_method=="pauli" and \
            sampler.quantum_instance.backend.name() == 'statevector_simulator':
            std = 0
        else:
            # this variance from qiskit = summed variance, not true variance!
            variance = np.real(expectation.compute_variance(sampled_expect_op))
            std = np.sqrt(variance / sampler.quantum_instance.run_config.shots)
    return np.array([mean, std])

def get_rand_init_pts(n, seed=None, norm=1e-1):
    np.random.seed(seed)
    return np.random.random(n)*norm

def get_eigenstate_from_params(ansatz, params, state, sampler=None):
    assert sampler == None or type(sampler) == CircuitSampler
    evolved_state = StateFn(state.compose(ansatz.bind_parameters(params)))
    if sampler == None:
        eval_state = evolved_state.eval()
    else:
        eval_state = sampler.convert(evolved_state).eval()
    return eval_state.primitive

def get_all_eigenstates_from_params(ansatz, params, state_list, sampler=None):
    return [get_eigenstate_from_params(ansatz, params, state, sampler=sampler) 
            for state in state_list]





# =======================================================================
#
# Following two classes work with IBM Quantum backends and noise model directly
# by constructing expectation circuits. They should work in a similar way.
# TODO: refactor and write a demo on them.
#
# =======================================================================

class SSVQE_ibmq_solver:
    
    def __init__(self, ansatz, pauli_sum_op, init_circs, ibmq_backend, init_pts, 
                 use_calibration=False, init_layout='default',
                 shots=8192, maxiter=100, sleep_interval=30, optimizer_name='SPSA'):
        # validation
        assert type(init_circs) == list
        assert all(circ.num_qubits == ansatz.num_qubits for circ in init_circs)
        assert ansatz.num_qubits == pauli_sum_op.num_qubits
        self.ibmq_qubits = ibmq_backend.configuration().num_qubits
        assert ansatz.num_qubits <= self.ibmq_qubits 
        
        self.pauli_sum_op = pauli_sum_op
        self.ansatz = ansatz
        self.init_circs = init_circs
        self.n_circ = len(init_circs)
        self.n_qubit = self.pauli_sum_op.num_qubits
        self.weights = np.array([1,0.5,0.25,0.125][:self.n_circ])
        assert self.n_circ == len(self.weights)
        self.ibmq_backend = ibmq_backend
        self.init_pts = init_pts
        self.shots = shots
        self.init_layout = init_layout if init_layout != 'default' else list(np.arange(self.n_qubit))
        assert max(self.init_layout) < self.ibmq_qubits
        self.job_id_hist = []
        self.all_eval_hist = []
        self.all_cost_hist = []
        self.all_param_hist = []
        self.cb_param_hist = []
        self.cb_cost_hist = []
        self.prepare_file_handlers()
        self.maxiter = maxiter
        if optimizer_name == 'SPSA':
            self.optimizer = SPSA(maxiter=self.maxiter, callback=self.callback_qiskit) #termination_checker=self.terminate_func)
        elif optimizer_name == "COBYLA":
            self.optimizer = COBYLA(maxiter=self.maxiter, callback=self.callback_scipy) 
        elif optimizer_name == 'QNSPSA':
            # Note this does not take into account of initial states
            fidelity = QNSPSA.get_fidelity(self.ansatz)
            self.optimizer = QNSPSA(fidelity, maxiter=self.maxiter, callback=self.callback_qiskit)
        self.sleep_interval = sleep_interval
        self.use_calibration = use_calibration
        
        if self.use_calibration:
            self.calibrator = MeasCalibration(
                self.ibmq_backend, list(range(self.n_qubit)),
                self.calib_job_id_hist_handler, 
                shots=self.shots, calib_interval_min=90)

   
    def extract_pauli_strs_and_coeffs(pauli_sum_op, ordered=False):
        pauli_strs, pauli_coeffs = [], []
        for i in range(len(pauli_sum_op)):
            pauli_list = pauli_sum_op[i].primitive.to_list()
            assert len(pauli_list) == 1
            s, c = pauli_list[0]
            assert c.imag == 0
            pauli_strs.append(s)
            pauli_coeffs.append(c.real)

        if ordered:
            ordered_list = [[pauli_strs[i], pauli_coeffs[i]] for i in range(len(pauli_strs))]
            ordered_list.sort(key=lambda x: -abs(x[1]))
            pauli_strs = [el[0] for el in ordered_list]
            pauli_coeffs = [el[1] for el in ordered_list]

        return pauli_strs, pauli_coeffs
    
    def prepare_post_meas_circs(self, ordered=True):
        self.pauli_str_list, self.pauli_coeff_list = SSVQE_ibmq_solver.extract_pauli_strs_and_coeffs(self.pauli_sum_op, ordered=ordered)
        self.post_meas_circs = []
        self.pauli_str_to_circuit_id = {}
        self.pauli_str_weight_dict = {}
        for pauli_str, coeff in zip(self.pauli_str_list, self.pauli_coeff_list):
            pauli_str_Z2I = pauli_str.replace('Z', 'I')
            self.pauli_str_weight_dict[pauli_str] = coeff
            if not pauli_str_Z2I in self.pauli_str_to_circuit_id:
                # print(pauli_str, pauli_str_Z2I)
                post_meas_circ = PostMeasurement.build(self.ansatz, pauli_str_Z2I)
                self.post_meas_circs.append(post_meas_circ)
                self.pauli_str_to_circuit_id[pauli_str_Z2I] = len(self.post_meas_circs) - 1
        print(f'Created a total of {len(self.post_meas_circs)} circuits for the Pauli sum operator.')
        return self.post_meas_circs
    
    def prepare_all_circs(self):
        self.prepare_post_meas_circs()
        self.final_circs = []
        self.final_circs_indexer = {} # map (init_circ_idx, post_meas_idx) => location of circ in final_circs
        for i, init_c in enumerate(self.init_circs):
            for j, post_meas_c in enumerate(self.post_meas_circs):
                self.final_circs.append(init_c.compose(post_meas_c))
                self.final_circs_indexer[(i, j)] = len(self.final_circs) - 1
        print(f'Created a total of {len(self.final_circs)} circuits for (initial circuits x Pauli sum operator) combinations.')        
        return self.final_circs
                
    def get_circ_id_by_init_and_pauli_sum(self, init_idx, pauli_str):
        post_meas_circ_id = self.pauli_str_to_circuit_id[pauli_str.replace('Z', 'I')]
        return self.final_circs_indexer[(init_idx, post_meas_circ_id)]
    
    def get_circ_directly_by_init_and_pauli_sum(self, init_idx, pauli_str):
        return self.final_circs[self.get_circ_id_by_init_and_pauli_sum(init_idx, pauli_str)]
    
    def prepare_file_handlers(self):
        self.time_stamp = datetime.now().strftime("%m_%d_%H_%M_%S")
        self.all_eval_hist_handler = FileIO(f"all_eval_hist_{self.time_stamp}.dat")
        self.all_cost_hist_handler = FileIO(f"all_cost_hist_{self.time_stamp}.dat")
        self.all_param_hist_handler = FileIO(f"all_param_hist_{self.time_stamp}.dat")
        self.cb_param_hist_handler = FileIO(f"cb_param_hist_{self.time_stamp}.dat")
        self.cb_cost_hist_handler = FileIO(f"cb_cost_hist_{self.time_stamp}.dat")
        self.ssvqe_job_id_hist_handler = FileIO(f"ssvqe_job_id_hist_{self.time_stamp}.dat")
        self.calib_job_id_hist_handler = FileIO(f"calib_job_id_hist_{self.time_stamp}.dat")
        print(f"Created all handlers at {self.time_stamp}.")
        return True
    
    def prepare_measurers(self):
        self.measurer_list = []
        self.measurer_dict = {}
        for pauli_str in self.pauli_str_list:
            meas = PauliMeasStrSum(pauli_str, self.pauli_str_weight_dict[pauli_str])
            self.measurer_list.append(meas)
            self.measurer_dict[pauli_str] = meas
        print(f"Created a total of {len(self.measurer_list)} Pauli-Measurers.")
        return self.measurer_list
            
    def get_expectation(self, counts):
        ''' counts is a list of histogram dictionary; return a list of expectation values '''
        
        total_count = sum(counts[0].values())

        def get_count(idx):
            assert idx < len(counts), "Count idx must be less than length of counts"
            return counts[idx]
        
        all_evals = []
        for i in range(len(self.init_circs)):
            s = 0
            for j, pauli_str in enumerate(self.pauli_str_list):
                circ_idx = self.get_circ_id_by_init_and_pauli_sum(i, pauli_str)
                data = get_count(circ_idx)
                s += self.measurer_dict[pauli_str].compute_with_count_dict(data)
            all_evals.append(s)
        
        return all_evals


    def transpile_circ_to_ibmq(self, circ):
        new_circ = QuantumCircuit(self.n_qubit, self.n_qubit)
        new_circ.append(circ, self.init_layout)
        new_circ.measure(self.init_layout, range(self.n_qubit))  # always measure to [0,1,...] classical bits
        return transpile(new_circ, self.ibmq_backend, initial_layout=self.init_layout)
    
    def get_cost(self, param):
                
        assert len(param) == self.final_circs[0].num_parameters
        
        binded_circuits = []
        print("Transpiling .... ", end='')
        for circ in self.final_circs:
            binded_circuits.append(self.transpile_circ_to_ibmq(circ.bind_parameters(param)))
        print("Done.")
        
        if self.job_id_hist != []: job_sleeper(self.sleep_interval) 
        job = execute(binded_circuits, self.ibmq_backend, shots=self.shots)
        jobID = job.job_id()
        self.ssvqe_job_id_hist_handler.append_text(jobID)
        self.job_id_hist.append(jobID)
        print("\n(Slept {:.0f}s) Job ID: {}".format(self.sleep_interval, jobID))
        job_monitor(job)
        
        result = job.result()
        if self.use_calibration:
            result = self.calibrator.apply_fitter_to(result)
        all_counts = result.get_counts()
    
        all_exps = self.get_expectation(all_counts)
        all_exps = np.array(all_exps)
        cost = self.weights.dot(all_exps)
        
        # File IO
        self.all_eval_hist.append(all_exps)
        self.all_eval_hist_handler.append_text(f"{all_exps.tolist()}")
        self.all_cost_hist.append(cost)
        self.all_cost_hist_handler.append_text(f"{cost}")
        self.all_param_hist.append(param)
        self.all_param_hist_handler.append_text(f"{param.tolist()}")
        
        print("| feval: {:>03} | {} | cost = {:.1f} | exp_vals = {} |\n".format(
            len(self.all_eval_hist), time_since(self.start_time), cost, np.round(all_exps, 1)))
        
        return cost

    def callback_qiskit(self, num_eval, param, f_eval, step, acc):
        ''' Callback function for SPSA, QNSPSA, etc'''
        # File IO
        self.cb_param_hist.append(param[:])
        self.cb_param_hist_handler.append_text(f"{param.tolist()}")
        self.cb_cost_hist.append(f_eval)
        self.cb_cost_hist_handler.append_text(f"{f_eval}")
        print("=============== | iter: {:>03} | cost = {:.2f} | ===============".format(len(self.cb_cost_hist), f_eval))

    def callback_scipy(self, param):
        self.cb_param_hist.append(param[:])
        self.cb_param_hist_handler.append_text(f"{param.tolist()}")
        cb_cost = self.all_cost_hist[-1]
        self.cb_cost_hist.append(cb_cost)
        self.cb_cost_hist_handler.append_text(f"{cb_cost}")
        print("=============== | iter: {:>03} | cost = {:.2f} | ===============".format(len(self.cb_cost_hist), cb_cost))

    # def terminate_func(self, num_eval, param, f_eval, step, acc):
    #     exact_cost = np.array(exact[:self.n_circ]).dot(self.weights[:self.n_circ])
    #     if abs(f_eval - exact_cost)/exact_cost*100 <= self.threshold:
    #         self.threshold_times += 1
    #         print(f'Met threshold {self.threshold_times} times.')
    #         if self.threshold_times == 5:
    #             # terminate when meet threshold a few times
    #             return True
    #     return False
            
    def run(self):
        self.start_time = time()
        print(f'SSVQE optimization started ({self.time_stamp})...')
        self.optimization_res = self.optimizer.minimize(fun=self.get_cost, x0=self.init_pts)
        print(self.optimization_res)
        print(f"Done. (data saved in: {self.time_stamp})")
        return self.optimization_res


class SSVQE_Noisy_solver:
    
    def __init__(self, ansatz, pauli_sum_op, init_circs, ibmq_backend, init_pts, 
                 noise_model, basis_gates, coupling_map,
                 use_calibration=False, init_layout='default',
                 shots=8192, maxiter=100, sleep_interval=30):
        # validation
        assert type(init_circs) == list
        assert all(circ.num_qubits == ansatz.num_qubits for circ in init_circs)
        assert ansatz.num_qubits == pauli_sum_op.num_qubits
        self.ibmq_qubits = ibmq_backend.configuration().num_qubits
        assert ansatz.num_qubits <= self.ibmq_qubits 
        
        self.pauli_sum_op = pauli_sum_op
        self.ansatz = ansatz
        self.init_circs = init_circs
        self.n_circ = len(init_circs)
        self.n_qubit = self.pauli_sum_op.num_qubits
        self.weights = np.array([1,0.5,0.25,0.125][:self.n_circ])
        assert self.n_circ == len(self.weights)
        self.ibmq_backend = ibmq_backend
        self.init_pts = init_pts
        self.noise_model = noise_model
        self.basis_gates = basis_gates
        self.coupling_map = coupling_map
        self.shots = shots
        self.init_layout = init_layout if init_layout != 'default' else list(np.arange(self.n_qubit))
        assert max(self.init_layout) < self.ibmq_qubits
        self.job_id_hist = []
        self.all_eval_hist = []
        self.all_cost_hist = []
        self.all_param_hist = []
        self.cb_param_hist = []
        self.cb_cost_hist = []
        self.prepare_file_handlers()
        self.maxiter = maxiter
        self.optimizer = SPSA(maxiter=self.maxiter, callback=self.callback_qiskit) #termination_checker=self.terminate_func)
        self.sleep_interval = sleep_interval
        self.use_calibration = use_calibration
        
        if self.use_calibration:
            self.calibrator = MeasCalibration(
                self.ibmq_backend, list(range(self.n_qubit)),
                self.calib_job_id_hist_handler, 
                shots=self.shots, calib_interval_min=90,
                noise_model=self.noise_model,
                basis_gates=self.basis_gates,
                coupling_map=self.coupling_map, override_qubits=True)

   
    def extract_pauli_strs_and_coeffs(pauli_sum_op, ordered=False):
        pauli_strs, pauli_coeffs = [], []
        for i in range(len(pauli_sum_op)):
            pauli_list = pauli_sum_op[i].primitive.to_list()
            assert len(pauli_list) == 1
            s, c = pauli_list[0]
            assert c.imag == 0
            pauli_strs.append(s)
            pauli_coeffs.append(c.real)

        if ordered:
            ordered_list = [[pauli_strs[i], pauli_coeffs[i]] for i in range(len(pauli_strs))]
            ordered_list.sort(key=lambda x: -abs(x[1]))
            pauli_strs = [el[0] for el in ordered_list]
            pauli_coeffs = [el[1] for el in ordered_list]

        return pauli_strs, pauli_coeffs
    
    def prepare_post_meas_circs(self, ordered=True):
        self.pauli_str_list, self.pauli_coeff_list = SSVQE_ibmq_solver.extract_pauli_strs_and_coeffs(self.pauli_sum_op, ordered=ordered)
        self.post_meas_circs = []
        self.pauli_str_to_circuit_id = {}
        self.pauli_str_weight_dict = {}
        for pauli_str, coeff in zip(self.pauli_str_list, self.pauli_coeff_list):
            pauli_str_Z2I = pauli_str.replace('Z', 'I')
            self.pauli_str_weight_dict[pauli_str] = coeff
            if not pauli_str_Z2I in self.pauli_str_to_circuit_id:
                # print(pauli_str, pauli_str_Z2I)
                post_meas_circ = PostMeasurement.build(self.ansatz, pauli_str_Z2I)
                self.post_meas_circs.append(post_meas_circ)
                self.pauli_str_to_circuit_id[pauli_str_Z2I] = len(self.post_meas_circs) - 1
        print(f'Created a total of {len(self.post_meas_circs)} circuits for the Pauli sum operator.')
        return self.post_meas_circs
    
    def prepare_all_circs(self):
        self.prepare_post_meas_circs()
        self.final_circs = []
        self.final_circs_indexer = {} # map (init_circ_idx, post_meas_idx) => location of circ in final_circs
        for i, init_c in enumerate(self.init_circs):
            for j, post_meas_c in enumerate(self.post_meas_circs):
                self.final_circs.append(init_c.compose(post_meas_c))
                self.final_circs_indexer[(i, j)] = len(self.final_circs) - 1
        print(f'Created a total of {len(self.final_circs)} circuits for (initial circuits x Pauli sum operator) combinations.')        
        return self.final_circs
                
    def get_circ_id_by_init_and_pauli_sum(self, init_idx, pauli_str):
        post_meas_circ_id = self.pauli_str_to_circuit_id[pauli_str.replace('Z', 'I')]
        return self.final_circs_indexer[(init_idx, post_meas_circ_id)]
    
    def get_circ_directly_by_init_and_pauli_sum(self, init_idx, pauli_str):
        return self.final_circs[self.get_circ_id_by_init_and_pauli_sum(init_idx, pauli_str)]
    
    def prepare_file_handlers(self):
        self.time_stamp = datetime.now().strftime("%m_%d_%H_%M_%S")
        self.all_eval_hist_handler = FileIO(f"all_eval_hist_{self.time_stamp}.dat")
        self.all_cost_hist_handler = FileIO(f"all_cost_hist_{self.time_stamp}.dat")
        self.all_param_hist_handler = FileIO(f"all_param_hist_{self.time_stamp}.dat")
        self.cb_param_hist_handler = FileIO(f"cb_param_hist_{self.time_stamp}.dat")
        self.cb_cost_hist_handler = FileIO(f"cb_cost_hist_{self.time_stamp}.dat")
        self.ssvqe_job_id_hist_handler = FileIO(f"ssvqe_job_id_hist_{self.time_stamp}.dat")
        self.calib_job_id_hist_handler = FileIO(f"calib_job_id_hist_{self.time_stamp}.dat")
        print(f"Created all handlers at {self.time_stamp}.")
        return True
    
    def prepare_measurers(self):
        self.measurer_list = []
        self.measurer_dict = {}
        for pauli_str in self.pauli_str_list:
            meas = PauliMeasStrSum(pauli_str, self.pauli_str_weight_dict[pauli_str])
            self.measurer_list.append(meas)
            self.measurer_dict[pauli_str] = meas
        print(f"Created a total of {len(self.measurer_list)} Pauli-Measurers.")
        return self.measurer_list
            
    def get_expectation(self, counts):
        ''' counts is a list of histogram dictionary; return a list of expectation values '''
        
        total_count = sum(counts[0].values())

        def get_count(idx):
            assert idx < len(counts), "Count idx must be less than length of counts"
            return counts[idx]
        
        all_evals = []
        for i in range(len(self.init_circs)):
            s = 0
            for j, pauli_str in enumerate(self.pauli_str_list):
                circ_idx = self.get_circ_id_by_init_and_pauli_sum(i, pauli_str)
                data = get_count(circ_idx)
                s += self.measurer_dict[pauli_str].compute_with_count_dict(data)
            all_evals.append(s)
        
        return all_evals


    def transpile_circ_to_ibmq(self, circ):
        new_circ = QuantumCircuit(self.n_qubit, self.n_qubit)
        new_circ.append(circ, self.init_layout)
        new_circ.measure(self.init_layout, range(self.n_qubit))  # always measure to [0,1,...] classical bits
        return transpile(new_circ, self.ibmq_backend, initial_layout=self.init_layout)
    
    def get_cost(self, param):
                
        assert len(param) == self.final_circs[0].num_parameters
        
        binded_circuits = []
        print("Transpiling .... ", end='')
        for circ in self.final_circs:
            binded_circuits.append(self.transpile_circ_to_ibmq(circ.bind_parameters(param)))
        print("Done.")
        
        if self.job_id_hist != []: job_sleeper(self.sleep_interval) 
        job = execute(binded_circuits, self.ibmq_backend, shots=self.shots,
                      coupling_map=self.coupling_map, basis_gates=self.basis_gates, 
                      noise_model=self.noise_model)
        jobID = job.job_id()
        self.ssvqe_job_id_hist_handler.append_text(jobID)
        self.job_id_hist.append(jobID)
        print("\n(Slept {:.0f}s) Job ID: {}".format(self.sleep_interval, jobID))
        job_monitor(job)
        
        result = job.result()
        if self.use_calibration:
            result = self.calibrator.apply_fitter_to(result)
        all_counts = result.get_counts()
    
        all_exps = self.get_expectation(all_counts)
        all_exps = np.array(all_exps)
        cost = self.weights.dot(all_exps)
        
        # File IO
        self.all_eval_hist.append(all_exps)
        self.all_eval_hist_handler.append_text(f"{all_exps.tolist()}")
        self.all_cost_hist.append(cost)
        self.all_cost_hist_handler.append_text(f"{cost}")
        self.all_param_hist.append(param)
        self.all_param_hist_handler.append_text(f"{param.tolist()}")
        
        print("| feval: {:>03} | {} | cost = {:.1f} | exp_vals = {} |\n".format(
            len(self.all_eval_hist), time_since(self.start_time), cost, np.round(all_exps, 1)))
        
        return cost

    def callback_qiskit(self, num_eval, param, f_eval, step, acc):
        ''' Callback function for SPSA, QNSPSA, etc'''
        # File IO
        self.cb_param_hist.append(param[:])
        self.cb_param_hist_handler.append_text(f"{param.tolist()}")
        self.cb_cost_hist.append(f_eval)
        self.cb_cost_hist_handler.append_text(f"{f_eval}")
        print("=============== | iter: {:>03} | cost = {:.2f} | ===============".format(len(self.cb_cost_hist), f_eval))


    # def terminate_func(self, num_eval, param, f_eval, step, acc):
    #     exact_cost = np.array(exact[:self.n_circ]).dot(self.weights[:self.n_circ])
    #     if abs(f_eval - exact_cost)/exact_cost*100 <= self.threshold:
    #         self.threshold_times += 1
    #         print(f'Met threshold {self.threshold_times} times.')
    #         if self.threshold_times == 5:
    #             # terminate when meet threshold a few times
    #             return True
    #     return False
            
    def run(self):
        self.start_time = time()
        print(f'SSVQE optimization started ({self.time_stamp})...')
        self.optimization_res = self.optimizer.minimize(fun=self.get_cost, x0=self.init_pts)
        print(self.optimization_res)
        print(f"Done. (data saved in: {self.time_stamp})")
        return self.optimization_res