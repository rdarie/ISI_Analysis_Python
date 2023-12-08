import numpy as np
import pandas as pd
from lmfit import Model, Parameters
def triphasic_ecap(
        t,
        amp1, amp2, amp3,
        wid1, wid2, wid3,
        t_offset, amp_offset):
    distance_factor = 1.
    offset2 = t_offset + distance_factor * (wid1 + wid2)
    offset3 = offset2 + distance_factor * (wid2 + wid3)
    peak1 = amp1 * np.exp(-((t - t_offset) / wid1) ** 2)
    peak2 = amp2 * np.exp(-((t - offset2) / wid2) ** 2)
    peak3 = amp3 * np.exp(-((t - offset3) / wid3) ** 2)
    return_val = amp_offset + peak1 + peak2 + peak3
    if np.isnan(return_val).any().any():
        print(f"NaNs in triphasic_ecap!")
    return return_val
def biphasic_ecap(
        t,
        amp1, amp2,
        wid1, wid2,
        t_offset1, t_offset2):
    offset1 = t_offset1
    offset2 = t_offset1 + t_offset2
    peak1 = amp1 * np.exp(-((t - offset1) / wid1) ** 2)
    peak2 = amp2 * np.exp(-((t - offset2) / wid2) ** 2)
    return_val = peak1 + peak2
    if np.isnan(return_val).any().any():
        print(f"NaNs in biphasic_ecap!")
    return return_val


def biexponential(t, amp1, tau1, amp2, tau2):
    return_val = amp1 * np.exp(-t / tau1) + amp2 * np.exp(-t / tau2)
    if np.isnan(return_val).any().any():
        print(f"NaNs in biexponential!")
    return return_val


def triexponential(t, amp1, tau1, amp2, tau2, amp3, tau3):
    return_val = amp1 * np.exp(-t / tau1) + amp2 * np.exp(-t / tau2) + amp3 * np.exp(-t / tau3)
    if np.isnan(return_val).any().any():
        print(f"NaNs in biexponential!")
    return return_val

def exponential(t, amp, tau):
    output = amp * np.exp(-t / tau)
    return output

def offset_exponential(t, amp, tau, offset):
    output = amp * np.exp(-t / tau) + offset
    if np.isnan(output).any().any():
        print(f"NaNs in offset_exponential!")
    return output

def gaussian(t, amp, wid, t_offset):
    output = amp * np.exp(-((t - t_offset) / wid) ** 2)
    return output

class ArtifactModel(object):

    def __init__(
            self,
            ecap_fun=triphasic_ecap, ecap_model=None,
            init_ecap_params=dict(),
            exp_fun=triexponential, exp_model=None,
            init_exp_params=dict(), fit_opts=dict(),
            num_macro_iterations=3, verbose=0):
        if ecap_model is not None:
            self.ecap_model = ecap_model
        else:
            self.ecap_model = Model(ecap_fun)
        if exp_model is not None:
            self.exp_model = exp_model
        else:
            self.exp_model = Model(exp_fun)
        self.ecap_params = init_ecap_params
        self.exp_params = init_exp_params
        self.exp_fit_history = []
        self.ecap_fit_history = []
        self.num_macro_iterations = num_macro_iterations
        self.fit_opts = fit_opts
        self.verbose = verbose

    def fit(self, t, y):
        ecap_residual = y.copy()
        self.exp_fit_history = []
        self.ecap_fit_history = []
        for i in range(self.num_macro_iterations):
            try:
                exp_result = self.exp_model.fit(
                    ecap_residual, self.exp_params, t=t, **self.fit_opts)
                if self.verbose > 1:
                    print(f'exp rqsuared = {exp_result.rsquared:.2f}')
                self.exp_fit_history.append(exp_result.rsquared)
                self.exp_params = exp_result.params
                exp_residual = y - self.exp_model.eval(t=t, params=self.exp_params)
                ecap_result = self.ecap_model.fit(
                    exp_residual, self.ecap_params, t=t, **self.fit_opts)
                if self.verbose > 1:
                    print(f'ecap rqsuared = {ecap_result.rsquared:.2f}')
                self.ecap_fit_history.append(ecap_result.rsquared)
                self.ecap_params = ecap_result.params
                ecap_residual = y - self.ecap_model.eval(t=t, params=self.ecap_params)
            except Exception as err_msg:
                print(f'{err_msg}')
                break
        if self.verbose > 0:
            print(f'ecap_params = {self.ecap_params.pretty_print()}')
            print(f'exp_params = {self.exp_params.pretty_print()}')

    def get_best_fit(self, t):
        best_fit = (
                self.exp_model.eval(params=self.exp_params, t=t) +
                self.ecap_model.eval(params=self.ecap_params, t=t))
        return best_fit
