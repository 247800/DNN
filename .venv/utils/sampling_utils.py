import torch
import torchaudio
import numpy as np
from utils.training_utils import load_state_dict

def get_time_schedule(sigma_min=1e-5, sigma_max=12, T=100, rho=10):
    i = torch.arange(0, T + 1)
    sigma_i = (sigma_max ** (1/rho) + i * (sigma_min ** (1/rho) - sigma_max ** (1/rho)) / (T - 1)) ** rho
    # print('sigma_i[-1]:', sigma_i[-1])
    # print('sigma_i[0]:', sigma_i[0])
    # print('sigma_i[T]:', sigma_i[T])
    # print('sigma_i:', sigma_i)
    sigma_i[T] = 0
    return sigma_i

def get_noise(t, S_tmin=1e-5, S_tmax=12, S_churn=None, idx=0):
    if S_churn is None:
        S_churn = t[idx:idx+1]
        # print('S_churn:', S_churn)
    if S_churn < S_tmin:
        gamma_i = 0
    elif S_churn > S_tmax:
        gamma_i = 0
    else:
        N = torch.randn(1)
        gamma_i = min(S_churn / N, np.sqrt(1) - 1)
    return gamma_i


def load_checkpoint(path, device, model):
    state_dict = torch.load(path, map_location=device, weights_only=False)
    try:
        it = state_dict["it"]
    except:
        it = 0
    print(f"loading checkpoint {it}")
    return load_state_dict(state_dict, ema=model)