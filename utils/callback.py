import torch
import matplotlib.pyplot as plt
import wandb
import soundfile as sf
import numpy as np
import os

def plot_time_schedule(t):
    plt.figure(figsize=(20, 6))
    plt.grid(True)
    plt.plot(t.t().numpy())
    plt.xlabel("Step")
    plt.ylabel("Sigma(t)")
    plt.title("Time Schedule")
    plt.savefig(f"time_schedule.pdf", format='pdf')
    plt.show()

# def prepare_folders():
    # if not os.path.exists("exp_wf"):
    #     os.makedirs("exp_wf")
    # if not os.path.exists("exp_spec"):
    #     os.makedirs("exp_spec")
    # if not os.path.exists("exp_wav"):
    #     os.makedirs("exp_wav")
    # if not os.path.exists("exp_img"):
    #     os.makedirs("exp_img")
    # if not os.path.exists("exp_loss"):
    #     os.makedirs("exp_loss")

def wandb_run(learning_rate, n_steps):
    wandb.login()
    wandb.init(
        project="dnn-sampler",
        config={
            "learning_rate": learning_rate,
            "steps": n_steps,
        },
    )

def export_waveform(x_hat, sample_rate, step, output_dir="exp_wf"):
    os.makedirs(output_dir, exist_ok=True)
    x_out = x_hat.squeeze().detach().cpu().numpy()
    time_axis = np.linspace(0, len(x_out) / sample_rate, num=len(x_out))

    plt.figure(figsize=(20, 6))
    plt.grid(True)
    plt.plot(time_axis,x_out)
    plt.xlabel("Čas [s]")
    plt.ylabel("Amplituda [-]")
    plt.title(f"Waveform {step + 1}")
    output_path = os.path.join(output_dir, f"wf_{step + 1}.pdf")
    plt.savefig(output_path, format='pdf')
    plt.show()

def export_spectrogram(x_hat, sample_rate, step, output_dir="exp_spec"):
    os.makedirs(output_dir, exist_ok=True)
    x_out = x_hat.detach().cpu().numpy()
    x_out_tensor = torch.from_numpy(x_out)
    hop_size = 256
    stft = lambda x: torch.stft(x.squeeze(1), n_fft=1024, hop_length=hop_size, win_length=1024, window=torch.hann_window(1024), return_complex=True)
    spec = stft(x_out_tensor)
    eps = 1e-8
    magnitude = torch.abs(spec)
    spectrogram_db = 20 * torch.log10(magnitude + eps)
    t = torch.linspace(0, (spec.shape[-1] - 1) * (hop_size / sample_rate), spec.shape[-1])
    freqs = torch.linspace(0, sample_rate // 2, 513)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, freqs, spectrogram_db[0].numpy(), shading='auto', vmin=spectrogram_db.min().item() + 50, vmax=spectrogram_db.max().item())
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Čas [s]")
    plt.ylabel("Frekvence [Hz]")
    plt.title(f"Spektrogram {step+1}")
    plt.savefig(f"exp_spec/spec_{step+1}.pdf", bbox_inches='tight')
    plt.show()

def export_audio(x, sample_rate, step, output_dir="exp_wav", dtype="float32"):
    os.makedirs(output_dir, exist_ok=True)
    x_out = x.squeeze().detach().cpu().numpy()
    x_out = x_out / np.max(np.abs(x_out))

    if dtype == "int16":
        x_out = (x_out * 32767).astype(np.int16)
        subtype = "PCM_16"
    elif dtype == "float32":
        x_out = x_out.astype(np.float32)
        subtype = "FLOAT"
    else:
        raise ValueError(f"Unsupported format: {dtype}. Use 'float32' or 'int16'.")

    filename = os.path.join(output_dir, f"export_{step + 1}.wav")
    sf.write(filename, x_out, sample_rate, subtype=subtype)
    print(f"Audio file exported: {filename} ({dtype})")



