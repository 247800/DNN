import torch
import torchaudio
import numpy as np
from utils.training_utils import load_state_dict
from utils.loss import l2_comp_stft_sum as rec_loss

def get_time_schedule(sigma_min=1e-5, sigma_max=12, T=100, rho=10):
    i = torch.arange(0, T + 1)
    sigma_i = (sigma_max ** (1/rho) + i * (sigma_min ** (1/rho) - sigma_max ** (1/rho)) / (T - 1)) ** rho
    sigma_i[T] = 0
    return sigma_i

def get_noise(t, S_tmin=1e-5, S_tmax=12, S_churn=None, idx=0):
    if S_churn is None:
        S_churn = t[idx:idx+1]
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

def get_likelihood_score(y, x_den, x, t, threshold):
    torch.cuda.empty_cache()
    y_hat = torch.clip(x_den, -threshold, threshold)
    rec = rec_loss(y.squeeze(1), y_hat.squeeze(1))
    loss = rec
    loss.backward(retain_graph=False)
    rec_grads = x.grad
    normguide = torch.norm(rec_grads) / ((x.shape[0] * x.shape[-1]) ** 0.5)
    zeta = 0.35
    zeta_hat = zeta / (normguide + 1e-8)
    return (-zeta_hat * rec_grads / t).detach(), rec

def get_preconditioning(sigma, sigma_data=0.063):
    c_skip = sigma_data**2 * (sigma**2 + sigma_data**2)**-1
    c_out = sigma*sigma_data * (sigma**2 + sigma_data**2)**(-0.5)
    c_in = (sigma_data**2 + sigma**2)**(-0.5)
    c_noise = (1/4)*torch.log(sigma)
    return c_skip, c_out, c_in, c_noise