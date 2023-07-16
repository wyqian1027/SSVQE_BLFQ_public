from time import time
from qiskit import execute, QuantumRegister
from qiskit.tools import job_monitor


from qiskit.utils.mitigation import (
    complete_meas_cal, tensored_meas_cal,
    CompleteMeasFitter, TensoredMeasFitter
)

class MeasCalibration:
    def __init__(self, ibmq_backend, qubit_list, file_handler, shots=8192, calib_interval_min=60,
                 noise_model=None, basis_gates=None, coupling_map=None, override_qubits=False):
        self.ibmq_backend = ibmq_backend
        self.shots = shots
        if override_qubits == False:
            self.n_qubit = self.ibmq_backend.configuration().num_qubits
        else:
            self.n_qubit = 7 # Nairobi
        self.file_handler = file_handler
        self.qr = QuantumRegister(self.n_qubit)
        self.qubit_list = qubit_list
        self.meas_calib_circuits, self.state_labels = complete_meas_cal(qubit_list=self.qubit_list, qr=self.qr, circlabel='mcal')
        self.calib_interval = calib_interval_min*60 # secs
        self.last_calib = float('-inf')
        self.noise_model = noise_model
        self.basis_gates = basis_gates
        self.coupling_map = coupling_map
    
    def get_meas_calib_circuits(self):
        return self.meas_calib_circuits
    
    def get_state_labels(self):
        return self.state_labels
    
    def execute_calib_circuits(self):
        if self.noise_model == None:
            self.job = execute(self.meas_calib_circuits, backend=self.ibmq_backend, shots=self.shots)
        else:
            self.job = execute(self.meas_calib_circuits, backend=self.ibmq_backend, shots=self.shots,
                               noise_model=self.noise_model, basis_gates=self.basis_gates, coupling_map=self.coupling_map)
        print("******* CALIBRATION IN PROGRESS *********")
        job_monitor(self.job)
        self.file_handler.append_text(self.job.job_id())
        self.last_calib = time()
        self.calib_result = self.job.result()
        
    def get_meas_fitter(self):
        self.meas_fitter = CompleteMeasFitter(
            results=self.calib_result, 
            state_labels=self.state_labels, 
            qubit_list=self.qubit_list, 
            circlabel='mcal')
        self.calib_matrix = self.meas_fitter.cal_matrix
        print(self.calib_matrix)
        return self.meas_fitter
    
    def apply_fitter_to(self, result):
        if (time() - self.last_calib) >= self.calib_interval:
            self.execute_calib_circuits()
            self.get_meas_fitter()
            
        mitigated_results = self.meas_fitter.filter.apply(result)
        mitigated_counts = mitigated_results.get_counts()
        return mitigated_results