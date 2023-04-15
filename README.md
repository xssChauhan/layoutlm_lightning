## layoutlm-lightning

PyTorch Lightning wrapper for LayoutLM family of models.


## Setup

1. Install `poetry`
2. Install dependencies using `poetry install`


## Training

`$ python scripts/train.py --data_dir <data_path>`


## TODO
- [ ] MLFLow / Logging integration
    - [x] Add logger
    - [x] Log training metrics
    - [ ] Log validation metrics per epoch
- Validation Loop
    - [x] Generate validation predictions per step as IOB
    - [x] Generate validation predictions per epoch 
- Inference, torchscript, quantization
- LayoutLMv2 
- LayoutLMv3