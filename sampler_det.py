import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset import AudioDataset
import hydra
import os
import torch.serialization
from omegaconf import ListConfig, DictConfig
from omegaconf.base import ContainerMetadata, Metadata
from typing import Any
from collections import defaultdict
from omegaconf.nodes import AnyNode
from utils.sampling_utils import load_checkpoint, get_preconditioning, get_time_schedule, get_likelihood_score
from utils import callback as cb

torch.serialization.add_safe_globals([
    ListConfig,
    ContainerMetadata,
    Any,
    list,
    defaultdict,
    dict,
    int,
    AnyNode,
    Metadata,
    DictConfig
])

os.environ['HYDRA_FULL_ERROR'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)

hydra.initialize(config_path="config", version_base="1.3")
cfg = hydra.compose(config_name="cqtdiff+_44k_32binsoct")
model = hydra.utils.instantiate(cfg)
weights = load_checkpoint(path="checkpoints/guitar_Career_44k_6s-325000.pt", device=device, model=model)

path = "IDMT-SMT-GUITAR"
sample_rate = 44100
dataset = AudioDataset(path, train=False, seg_len=262144)
dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

# S_noise = 1
n_steps = 100
sigma_min = 1e-6
sigma_max = 10
rho = 7

t = get_time_schedule(sigma_min=sigma_min, sigma_max=sigma_max, T=n_steps, rho=rho)
# cb.plot_time_schedule(t)

model.eval()
model.to(device)

n_sigs = 10
# thresholds = np.arange(0.05, 0.15, 0.05)
thresholds = np.arange(0.05, 0.51, 0.05)

for i in range(n_sigs):
    for threshold in thresholds:
        print(f"Input: {i+1}, Threshold: {threshold}")
        input_sig = next(iter(dataloader))
        torchaudio.save("x_inp{i+1}.wav", input_sig.squeeze(0).cpu(), sample_rate)

        x_i = torch.randn(input_sig.shape, device=device) * sigma_max
        y = torch.clip(input_sig, -threshold, threshold).to(device).requires_grad_(True)

        for step in range(n_steps):
               c_skip, c_out, c_in, c_noise = get_preconditioning(t[step])
               x_i = x_i.requires_grad_()
               x_den = c_skip * x_i + c_out * model((c_in * x_i).to(torch.float32), c_noise.to(torch.float32)).to(x_i.dtype)  # Get tweedie
               x_den = model.CQTransform.apply_hpf_DC(x_den)
               score_unc = (x_den - x_i) / (t[step]**2)  # Calculate score
               likelihood_score, _ = get_likelihood_score(y=y, x_den=x_den, x=x_i, t=t[step], threshold=threshold)
               d = score_unc + likelihood_score
               ode_integrant = d * -t[step]
               x_i = x_i + (t[step+1] - t[step]) * ode_integrant # Euler step

               # Detach from computational graph
               x_i = x_i.detach()
               x_den  = x_den.detach()

               print(f"Step: {step+1}, dt {t[step+1] - t[step]},"
                     f"t, {t[step].numpy()}, norm: {torch.norm(x_i - x_den).item()}")

        cb.export_audio(x_hat=x_den, sample_rate=sample_rate, i=i, threshold=threshold)
        cb.export_waveform(x_hat=x_den, sample_rate=sample_rate, i=i, threshold=threshold)
        cb.export_spectrogram(x_hat=x_den, sample_rate=sample_rate, i=i, threshold=threshold)
        torchaudio.save(f"x_out_{i+1}_thr{threshold}.wav", x_i.squeeze(0).cpu(), sample_rate)
