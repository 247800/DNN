import torch

def get_frequency_weighting(freqs, freq_weighting=None):
    if freq_weighting is None:
        return torch.ones_like(freqs).to(freqs.device)
    elif freq_weighting == "sqrt":
        return torch.sqrt(freqs)
    elif freq_weighting == "exp":
        freqs = torch.exp(freqs)
        return freqs - freqs[:, 0, :].unsqueeze(-2)
    elif freq_weighting == "log":
        return torch.log(1 + freqs)
    elif freq_weighting == "linear":
        return freqs

def l2_comp_stft_sum(x, x_hat):
    win_length = 1024
    fft_size = 2048
    hop_size = 256
    window = torch.hann_window(win_length).float().to(x.device)
    X = torch.stft(
        x,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    X_hat = torch.stft(
        x_hat,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    freqs = (
            torch.linspace(0, 1, X.shape[-2])
            .to(X.device)
            .unsqueeze(-1)
            .unsqueeze(0)
            .expand(X.shape)
            + 1
    )
    X = X * freqs
    X_hat = X_hat * freqs
    compression_factor = 0.667
    X_comp = (X.abs() + 1e-8) ** compression_factor * torch.exp(
        1j * X.angle()
    )
    X_hat_comp = (X_hat.abs() + 1e-8) ** compression_factor * torch.exp(
        1j * X_hat.angle()
    )
    loss = torch.sum((X_comp - X_hat_comp).abs() ** 2)
    weight = 1.0
    return weight * loss

