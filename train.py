import torch
import torchaudio
import torch.optim as optim
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from dataset import AudioDataset
from utils.loss import l2_comp_stft_sum as loss_fn
# from CQT.unet_octCQT import Unet_octCQT
from unet_octCQT import Unet_octCQT
import utils.cqt_nsgt_pytorch.CQT_nsgt
# from model import Denoiser
# import matplotlib.pyplot as plt
# from torchaudio.transforms import Spectrogram
# from torch.optim.lr_scheduler import _LRScheduler
# import auraloss
# import wandb
# wandb.login()
import hydra
from omegaconf import DictConfig
import os

os.environ['HYDRA_FULL_ERROR'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)

hydra.initialize(config_path="config", version_base="1.4")  # Set correct path
cfg = hydra.compose(config_name="cqtdiff+_44k_32binsoct")  # Use correct config name
model = hydra.utils.instantiate(cfg)
# model.eval()  # Set to eval mode
# model.to(device)

# path = "./GTZAN_dataset/"
path = "GTZAN_dataset"
dataset = AudioDataset(path, train=True)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# model = Denoiser()
# model = Unet_octCQT(cfg)
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

# loss_func = nn.MSELoss()
# loss_func = auraloss.freq.MultiResolutionSTFTLoss()
# loss_func = loss_fn()
# loss_fn = l2_comp_stft_sum()

learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
n_epochs = 5
epoch = 0
train_loss = []

# run = wandb.init(
#     # Set the project where this run will be logged
#     project="dnn-train",
#     # Track hyperparameters and run metadata
#     config={
#         "learning_rate": learning_rate,
#         "epochs": n_epochs,
#     },
# )

model.train()
model.to(device)

for epoch in range(n_epochs):
    for index, input_sig in enumerate(dataloader):
        waveform = input_sig
        print("input:", waveform)
        print("input shape:", waveform.shape)
        # print(waveform)
        # waveform = torch.unsqueeze(waveform, 1)
        # noise = 10
        # corrupted_sig = waveform + noise * torch.randn(waveform.shape)
        corrupted_sig = waveform + torch.randn(waveform.shape)
        print("corrupted sig:", corrupted_sig)
        print("corrupted sig shape:", corrupted_sig.shape)

        optimizer.zero_grad()
        # output = model(corrupted_sig)
        output = model(waveform,corrupted_sig)

        # loss = loss_func(output, waveform).to(device)
        # loss = loss_fn(x=x, x_hat=x_hat).to(device)
        loss = loss_fn(x=waveform.squeeze(1), x_hat=output.squeeze(1)).to(device)
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()
        train_loss.append(loss.item())

        print(f"Epoch: [{epoch+1}/{n_epochs}], Step: {index+1}, Loss: {loss}")
        # wandb.log({"loss": loss})

# torch.save(model.state_dict(), 'model_weights.pth')
# torch.save(model.state_dict(), 'denoiser.pt')
# torch.save(model.state_dict(), 'unet.pt')