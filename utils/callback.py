import torch
from torchaudio.transforms import Spectrogram
import matplotlib.pyplot as plt
import wandb
from scipy.io import wavfile
from scipy.io.wavfile import write
import numpy as np

def wandb_run(learning_rate, n_epochs):
    wandb.login()
    wandb.init(
        project="dnn-sampler",
        config={
            "learning_rate": learning_rate,
            "epochs": n_epochs,
        },
    )

def export_waveform(x_hat, step):
    x_out = x_hat.squeeze().detach().cpu().numpy()
    x_out_tensor = torch.from_numpy(x_out)
    # x_out_tensor = x_out_tensor[0,0, 0:44100]
    x_squeezed = x_out_tensor.squeeze()
    plt.figure(figsize=(20, 6))
    plt.grid(True)
    plt.plot(x_out)
    plt.xlabel("Čas [s]")
    plt.ylabel("Amplituda")
    plt.title(f"Raw Waveform {step + 1}")
    plt.savefig(f"exp_wf/wf_{step + 1}.pdf", format='pdf')
    plt.show()

def export_spectrogram(x_hat, sample_rate):
    x_out = x_hat.squeeze().detach().cpu().numpy()
    x_out_tensor = torch.from_numpy(x_out)
    # print(x_out_tensor.shape)
    # x_out_tensor = x_out_tensor[:, 0:44100]
    # print("sliced", x_out_tensor.shape)
    # x_squeezed = x_out_tensor.squeeze()
    # print("squeezed", x_squeezed.shape)
    hop_size = 256
    stft = lambda x: torch.stft(x, n_fft=1024, hop_length=hop_size, win_length=1024, window=torch.hann_window(1024), return_complex=True)
    freqs = torch.linspace(0, sample_rate // 2, 513)  # 1024//2 + 1
    specgram = stft(x_out_tensor)
    eps = 1e-8
    spectrogram_db = 20 * torch.log10(torch.abs(specgram) + eps)
    t = torch.linspace(0, (specgram.shape[-1] - 1) * (hop_size / sample_rate), specgram.shape[-1])
    # spectrogram = Spectrogram()(waveform)
    # spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
    print("Spectrogram shape: ", spectrogram_db.shape, t.shape)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, freqs, spectrogram_db[0].numpy(), vmin=spectrogram_db.min().item() + 50, vmax=spectrogram_db.max().item())
    # # plt.colorbar()
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Čas [s]")
    plt.ylabel("Frekvence [Hz]")
    plt.title("Spectrogram")
    plt.savefig(f"exp_spec/spec_{step+1}.pdf", bbox_inches='tight')

def export_audio(x_hat, sample_rate, step):
    x_out = x_hat.squeeze().detach().cpu().numpy()
    print(x_out.shape)
    x_out = x_out / np.max(np.abs(x_out))  # normalize to [-1, 1]
    x_i_int16 = (x_out * 32767).astype(np.int16)
    # x_i_int16 = x_i_int16.squeeze()
    # # print(type(x_i_int16), x_i_int16.dtype, x_i_int16.shape)
    write(f"exp_wav/export_{step+1}.wav", sample_rate, x_i_int16)
    # torchaudio.save("exports/export_1.wav", x_i_int16, sample_rate=22050)
    # torchaudio.save(f"exports/export_step_{step+1}.wav", x_i[step].cpu().numpy(), sample_rate=22050)
    print("audio file exported")

def plot_time_schedule(t):
    plt.figure(figsize=(20, 6))
    plt.grid(True)
    plt.plot(t.t().numpy())
    plt.xlabel("Step")
    plt.ylabel("Sigma(t)")
    plt.title("Time Schedule")
    plt.show()


