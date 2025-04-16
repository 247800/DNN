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

# torch.serialization.add_safe_globals([ListConfig])
# torch.serialization.add_safe_globals([ContainerMetadata])
# torch.serialization.add_safe_globals([Any])
# torch.serialization.add_safe_globals([list])
# torch.serialization.add_safe_globals([defaultdict])
# torch.serialization.add_safe_globals([dict])
# torch.serialization.add_safe_globals([int])
# torch.serialization.add_safe_globals([AnyNode])
# torch.serialization.add_safe_globals([Metadata])
# torch.serialization.add_safe_globals([DictConfig])

os.environ['HYDRA_FULL_ERROR'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)

hydra.initialize(config_path="config", version_base="1.4")  # Set correct path
cfg = hydra.compose(config_name="cqtdiff+_44k_32binsoct")  # Use correct config name
model = hydra.utils.instantiate(cfg)
# print("model", model)
weights = load_checkpoint(path="checkpoints/guitar_Career_44k_6s-325000.pt", device=device, model=model)
# print(type(weights))
# weights = torch.load("checkpoints/guitar_Career_44k_6s-325000.pt", weights_only=True)
# weights = EMAWarmup.load_checkpoint(path="checkpoints/guitar_Career_44k_6s-325000.pt")
# print("weights keys", weights.keys())
# model.load_state_dict(weights, strict=False)

path = "IDMT-SMT-GUITAR"
# path = "GTZAN_dataset"
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
# loss_func = loss()
# test_loss = []

# n_steps = 50
n_steps = 100
S_noise = 1
t = s_utils.get_time_schedule()
# print("t device", t.device)

# metriky

# plt.figure(figsize=(20, 6))
# plt.grid(True)
# plt.plot(t.t().numpy())
# plt.xlabel("Step")
# plt.ylabel("Sigma(t)")
# plt.title("Time Schedule")
# plt.show()

model.eval()
model.to(device)

learning_rate = 0.001
n_epochs = 5
# run = wandb.init(
#     # Set the project where this run will be logged
#     project="dnn-sampler",
#     # Track hyperparameters and run metadata
#     config={
#         "learning_rate": learning_rate,
#         "epochs": n_epochs,
#     },
# )

input_sig = next(iter(dataloader))
# x_i = input_sig.to(device)
x_i = torch.randn(input_sig.shape, device=device)
# x_0 = x + torch.randn(x.shape, device=device)

for step in range(n_steps):
    with torch.no_grad():
        gamma = s_utils.get_noise(t=t, idx=step)
        t_hat = t[step] + t[step] * gamma
        t_hat = t_hat.to(device)
        noise = torch.randn(x_i.shape, device=device)
        x_hat = x_i + torch.sqrt(t_hat.clone().detach() ** 2 - t[step] ** 2) * noise
        D_theta = model(x_hat, sigma=t[step])
        d = (D_theta - x_hat) / t_hat
        x_i = x_hat + (t[step + 1] - t_hat) * d
        # print(x_i.size, x_hat.size)
        print(x_i.shape, x_hat.shape)
        print("t",t[step])

        if step % 5 == 0:

            # Export wav file
            x_out = x_hat.squeeze().detach().cpu().numpy()
            print(x_out.shape)
            x_out = x_out / np.max(np.abs(x_out))  # normalize to [-1, 1]
            x_i_int16 = (x_out * 32767).astype(np.int16)
            # x_i_int16 = x_i_int16.squeeze()
            # # print(type(x_i_int16), x_i_int16.dtype, x_i_int16.shape)
            # write(f"exp_wav/export_{step+1}.wav", 22050, x_i_int16)
            # torchaudio.save("exports/export_1.wav", x_i_int16, sample_rate=22050)
            # torchaudio.save(f"exports/export_step_{step+1}.wav", x_i[step].cpu().numpy(), sample_rate=22050)

            # Export waveform
            x_out_tensor = torch.from_numpy(x_out)
            # x_out_tensor = x_out_tensor[0,0, 0:44100]
            x_squeezed = x_out_tensor.squeeze()
            plt.figure(figsize=(20, 6))
            plt.grid(True)
            plt.plot(x_out)
            plt.xlabel("Čas [s]")
            plt.ylabel("Amplituda")
            plt.title(f"Raw Waveform {step+1}")
            plt.savefig(f"exp_wf/wf_{step + 1}.pdf", format='pdf')
            plt.show()

        # Export spectrogram
        # sample_rate = 44100
        # # x_out_tensor = torch.from_numpy(x_out)
        # # print(x_out_tensor.shape)
        # # x_out_tensor = x_out_tensor[:, 0:44100]
        # # print("sliced", x_out_tensor.shape)
        # # x_squeezed = x_out_tensor.squeeze()
        # # print("squeezed", x_squeezed.shape)
        # hop_size = 256
        # stft = lambda x: torch.stft(x, n_fft=1024, hop_length=hop_size, win_length=1024, window=torch.hann_window(1024), return_complex=True)
        # freqs = torch.linspace(0, sample_rate // 2, 513)  # 1024//2 + 1
        # specgram = stft(x_out_tensor)
        # eps = 1e-8  # for numerical stability, nechcem log(0)
        # spectrogram_db = 20 * torch.log10(torch.abs(specgram) + eps)
        # t = torch.linspace(0, (specgram.shape[-1] - 1) * (hop_size / sample_rate), specgram.shape[-1])
        # # spectrogram = Spectrogram()(waveform)
        # # spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
        # print("Spectrogram shape: ", spectrogram_db.shape, t.shape)
        # plt.figure(figsize=(10, 6))
        # plt.pcolormesh(t, freqs, spectrogram_db[0].numpy(), vmin=spectrogram_db.min().item() + 50, vmax=spectrogram_db.max().item())
        # # # plt.colorbar()
        # plt.colorbar(format="%+2.0f dB")
        # plt.xlabel("Čas [s]")
        # plt.ylabel("Frekvence [Hz]")
        # plt.title("Spectrogram")
        # plt.savefig(f"exp_spec/spec_{step+1}.pdf", bbox_inches='tight')



        # loss = loss_fn(x=x, x_hat=x_i).to(device)
        # loss = loss_fn(x=x.squeeze(1), x_hat=x_i.squeeze(1)).to(device)

        print(f"Step: {step + 1}")
        # print(f"Step: [{step + 1}/{n_steps}], Loss: {loss}")
        # wandb.log({"loss": loss})

        # metriky.step
        # print(f"Step: [{step+1}/{n_steps}], Step: {index+1}, Loss: {loss}")
        # print("Loss device:", loss.device)

# wandb.finish()
