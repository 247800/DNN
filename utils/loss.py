import torch
import torchaudio
# from librosa.filters import mel as librosa_mel_fn
# from auraloss.freq import MultiResolutionSTFTLoss, RandomResolutionSTFTLoss

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

# def l2_comp_stft_sum(loss_args, x, x_hat):
def l2_comp_stft_sum(x, x_hat):
    win_length = 1024
    fft_size = 2048
    hop_size = 256
    # win_length = loss_args.get("win_length", 1024)
    # fft_size = loss_args.get("fft_size", 2048)
    # hop_size = loss_args.get("hop_size", 256)
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
    # freqs = get_frequency_weighting(
    #     freqs, freq_weighting=loss_args.get("freq_weighting", None)
    #     )

    X = X * freqs
    X_hat = X_hat * freqs

    # compression_factor = loss_args.get("compression_factor", None)
    # assert (
    #        compression_factor is not None
    #        and compression_factor > 0.0
    #        and compression_factor <= 1.0
    # ), f"Compression factor weird: {compression_factor}"
    compression_factor = 1
    X_comp = (X.abs() + 1e-8) ** compression_factor * torch.exp(
        1j * X.angle()
    )
    X_hat_comp = (X_hat.abs() + 1e-8) ** compression_factor * torch.exp(
        1j * X_hat.angle()
    )
    loss = torch.sum((X_comp - X_hat_comp).abs() ** 2)

    # else:
    #    raise NotImplementedError(
    #        f"rec_loss {loss_args.name} not implemented"
    #    )

    # weight = loss_args.get("weight", 1.0)
    weight = 1.0

    return weight * loss

    # return lambda x, x_hat: loss_fn(x, x_hat)
