import torch
import torchaudio
import torch.optim as optim
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from dataset import AudioDataset
# from model import Denoiser
from unet_octCQT import Unet_octCQT
import utils.cqt_nsgt_pytorch.CQT_nsgt
import utils.sampling_utils as s_utils
from utils.loss import l2_comp_stft_sum as loss_fn
# import auraloss
from torchaudio.transforms import Spectrogram
import matplotlib.pyplot as plt
# import wandb
# wandb.login()
import hydra
import os
import torch.serialization
from omegaconf import ListConfig, DictConfig
from omegaconf.base import ContainerMetadata, Metadata
from typing import Any
from collections import defaultdict
from omegaconf.nodes import AnyNode
from scipy.io import wavfile
from scipy.io.wavfile import write
from utils.sampling_utils import load_checkpoint
from utils.sampling_utils import get_time_schedule
from utils.sampling_utils import get_likelihood_score
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

hydra.initialize(config_path="config", version_base="1.4")
cfg = hydra.compose(config_name="cqtdiff+_44k_32binsoct")
model = hydra.utils.instantiate(cfg)
weights = load_checkpoint(path="checkpoints/guitar_Career_44k_6s-325000.pt", device=device, model=model)
# weights = torch.load("checkpoints/guitar_Career_44k_6s-325000.pt", weights_only=True)
# weights = EMAWarmup.load_checkpoint(path="checkpoints/guitar_Career_44k_6s-325000.pt")
# print("weights keys", weights.keys())
# model.load_state_dict(weights, strict=False)

path = "IDMT-SMT-GUITAR"
# path = "GTZAN_dataset"
sample_rate = 44100
# sample_rate = 22050
dataset = AudioDataset(path, train=False, seg_len=262144)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# model = Denoiser()
# model.load_state_dict(torch.load('denoiser.pt', weights_only=True))
# model = Unet_octCQT(
# 	depth=8,
#     emb_dim=256,
#     Ns=[32,32, 64 ,64, 128, 128,256, 256],
#     attention_layers=[0, 0, 0, 0, 0, 0, 0, 0],
#     checkpointing=[True, True, True, True, True, True, True, True],
#     Ss=[2,2,2,2, 2, 2, 2, 2],
#     num_dils=[1,3,4,5,5,6,6,7],
#     cqt = {
#         "window": "kaiser",
#     	"beta": 1,
#     	"num_octs": 8,
# 		"bins_per_oct": 32,
#     },
#     bottleneck_type="res_dil_convs",
#     num_bottleneck_layers=1,
#     attention_dict = {
# 	    "num_heads": 8,
#         "attn_dropout": 0.0,
#     	"bias_qkv": False,
# 		"N": 0,
#     	"rel_pos_num_buckets": 32,
#     	"rel_pos_max_distance": 64,
# 		"use_rel_pos": True,
#    		"Nproj": 8
#     })

# model = Unet_octCQT({
#     depth: 8,
#     emb_dim: 256,
#     Ns: [32, 32, 64, 64, 128, 128, 256, 256],
#     attention_layers: [0, 0, 0, 0, 0, 0, 0, 0],
#     checkpointing: [True, True, True, True, True, True, True, True],
#     Ss: [2, 2, 2, 2, 2, 2, 2, 2],
#     num_dils: [1, 3, 4, 5, 5, 6, 6, 7],
#     sample_rate: 44100,
#     audio_len: 262144,
#     cqt: {
#         window: "kaiser",
#         beta: 1,
#         num_octs: 8,
#         bins_per_oct: 32,
#     },
#     bottleneck_type: "res_dil_convs",
#     num_bottleneck_layers: 1,
#     attention_dict: {
#         num_heads: 8,
#         attn_dropout: 0.0},
#     bias_qkv: False,
#     N: 0,
#     rel_pos_num_buckets: 32,
#     rel_pos_max_distance: 64,
#     use_rel_pos: True,
#     Nproj: 8
# })

# model = Unet_octCQT({
#     "depth": 8,
#     "emb_dim": 256,
#     "Ns": [32, 32, 64, 64, 128, 128, 256, 256],
#     "attention_layers": [0, 0, 0, 0, 0, 0, 0, 0],
#     "checkpointing": [True, True, True, True, True, True, True, True],
#     "Ss": [2, 2, 2, 2, 2, 2, 2, 2],
#     "num_dils": [1, 3, 4, 5, 5, 6, 6, 7],
#     "sample_rate": 44100,
#     "audio_len": 262144,
#     "cqt": {
#         "window": "kaiser",
#         "beta": 1,
#         "num_octs": 8,
#         "bins_per_oct": 32,
#     },
#     "bottleneck_type": "res_dil_convs",
#     "num_bottleneck_layers": 1,
#     "attention_dict": {
#         "num_heads": 8,
#         "attn_dropout": 0.0
#     },
#     "bias_qkv": False,
#     "N": 0,
#     "rel_pos_num_buckets": 32,
#     "rel_pos_max_distance": 64,
#     "use_rel_pos": True,
#     "Nproj": 8
# })

# loss_func = auraloss.freq.MultiResolutionSTFTLoss()

learning_rate = 0.001
S_noise = 1
n_steps= 50
t= get_time_schedule(sigma_min=1e-5, sigma_max=1, T=n_steps, rho=10)

model.eval()
model.to(device)

# cb.plot_time_schedule(t)
# cb.wandb_run(learning_rate, n_epochs)

input_sig = next(iter(dataloader))
# x_i = input_sig.to(device)
x_i = torch.randn(input_sig.shape, device=device)
# x_0 = x + torch.randn(x.shape, device=device)

for step in range(n_steps):
   with torch.no_grad():
       x_hat = model(x_i, t[step])  # Get denoised estimate
       score_unc = (x_i - x_hat) / t[step]  # Calculate score
       threshold = 0.05
       y = torch.clip(input_sig, -threshold, threshold)
       # likelihood_score = get_likelihood_score(y=y, x_den=x_hat, x=x_i, t=t, threshold=threshold)
       # d = score_unc + likelihood_score
       d = (x_i - x_hat) / t[step]
       x_i = x_i + (t[step + 1] - t[step]) * d  # Euler step

       if step % 5 == 0:
           cb.export_audio(x_hat, sample_rate, step)
           cb.export_waveform(x_hat, step)
           # cb.export_spectrogram(x_hat, sample_rate)

       print(f"Step: {step + 1}")
       print("dt", t[step + 1] - t[step],"t",t[step].numpy())
       print("NORM", torch.norm(x_i - x_hat).item())
       # wandb.log({"loss": loss})
       # print("Loss device:", loss.device)

torchaudio.save("x_den.wav", x_i.squeeze(0).cpu(), sample_rate)


# wandb.finish()
