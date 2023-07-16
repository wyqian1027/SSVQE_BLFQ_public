from visualization import plot_param_list, plot_exp_val_list
import pickle
import numpy as np
from datetime import datetime


class OptimizationResult:

    def __init__(self, filename=None, optimizer_name=""):
        ''' Convenience result class for storing, loading, saving, and plotting multiple optimization results '''
        self.vqe_result = []
        self.optimizer_name = optimizer_name
        self.time = []
        self.final_param = []
        self.param_hist = []
        self.eval_hist = []
        self.cost_hist = []
        self.error_hist = []
        self.final_eval = []
        self.n_iter = []
        self.filename = filename
        if filename != None: self.load(filename)

    def __len__(self):
        return len(self.param_hist)

    def add_result(self, param_hist, eval_hist, cost_hist, error_hist, final_param, final_eval, time, vqe_result=None):
        self.param_hist.append(np.array(param_hist))
        self.eval_hist.append(np.array(eval_hist))
        self.cost_hist.append(np.array(cost_hist))
        self.error_hist.append(np.array(error_hist))
        self.final_eval.append(final_eval)
        self.n_iter.append(len(param_hist))
        self.final_param.append(final_param)
        self.time.append(time)
        self.vqe_result.append(vqe_result)


    def plot_params(self, idx=0):
        return plot_param_list(self.param_hist[idx])

    def plot_evals(self, idx=0, show_err=True, labels=[], ref_evals=[], fs=15, figsize=(8,6), legend_loc='best'):
        eval_data = [self.eval_hist[idx][:,i].reshape(-1,1) for i in range(len(self.eval_hist[idx][0]))]
        err_data = [self.error_hist[idx][:,i].reshape(-1,1) for i in range(len(self.error_hist[idx][0]))] if show_err else []
        plot_exp_val_list(eval_data, err_data, labels=labels, ref_evals=ref_evals, 
                          fs=fs, figsize=figsize, legend_loc=legend_loc)

    def plot_costs(self, idxs=[0], show_err=True, fs=15, figsize=(8,6), legend_loc='best'):
        err_data = [self.error_hist[idx] for idx in idxs] if show_err else []
        plot_exp_val_list([self.cost_hist[idx] for idx in idxs], err_data, 
                          fs=fs, figsize=figsize, legend_loc=legend_loc)

    

        


    def load(self, filename):
        self.filename = filename
        with open(self.filename, 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict) 
        print(f"OptimizationResult loaded.")
        return self

    def save(self, new_filename, insert_stamp=True):
        if insert_stamp == True:
            new_filename = f"{new_filename}_{get_time_stamp()}.pk"
        else:
            new_filename = f"{new_filename}.pk"
        with open(new_filename, 'wb') as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"OptimizationResult saved to {new_filename}.")
        return new_filename



def get_time_stamp():
    # return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return datetime.now().strftime("%m_%d_%H_%M_%S")


