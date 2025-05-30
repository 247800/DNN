# Diffusion model neural networks for restoration of damaged audio

Welcome to the repositary of my bachelor thesis. Here you can find the code for a deterministic diffusion posterior sampler I created for the 
[NLDistortionDiff](https://github.com/michalsvento/NLDistortionDiff) model.

> PODESTÁTOVÁ, Věra. Diffusion model neural networks for restoration of damaged audio. Online, bachelor's Thesis. Michal ŠVENTO (supervisor). Brno: Brno University of Technology, Faculty of Electrical Engineering and Communication, 2025. Available at: https://www.vut.cz/en/students/final-thesis/detail/167399.

## Setup

This code was tested with Python 3.10 and PyTorch 2.6.

To install the requirements:
```
pip install -r requirements.txt
```

It is also necessary to download the model, weights and checkpoint from the [NLDistortionDiff](https://github.com/michalsvento/NLDistortionDiff):

- model: [`unet_octCQT.py`](https://github.com/michalsvento/NLDistortionDiff/blob/6241780586e1ebd1ec6f67e03651c3976060aa6e/networks/unet_octCQT.py)
- utils: [`cqt_nsgt_pytorch`](https://github.com/michalsvento/NLDistortionDiff/tree/6241780586e1ebd1ec6f67e03651c3976060aa6e/utils/cqt_nsgt_pytorch)
- weights: [`cqtdiff+_44k_32binsoct.yaml`](https://github.com/michalsvento/NLDistortionDiff/blob/6241780586e1ebd1ec6f67e03651c3976060aa6e/conf/network/cqtdiff%2B_44k_32binsoct.yaml)

To get checkpoint, download [`checkpoints.zip`](https://github.com/michalsvento/NLDistortionDiff/releases/tag/checkpoints). 
The checkpoint used for this code was `guitar_Career_44k_6s-325000.pt`

It is recommended to include the `cqt_nsgt_pytorch` folder into the `utils` directory of this project. 

Before running this code, make sure to set the path to the weights, checkpoints and dataset in the `sampler_det.py`. The dataset used for this code was [IDMT-SMT-Guitar](https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html).

## Inference
For inference run:

```
sampler_det.py
```
