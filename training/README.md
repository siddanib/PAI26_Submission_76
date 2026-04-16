# Training

This folder contains the training and export utilities for the flow-based model used in this project.

## Verified entry points

- `train.py` trains the model.
  - It is configured with [Hydra](./conf/config.yaml), and the default config overrides `hydra/launcher` with `joblib`.
  - Training outputs are written to the Hydra run directory and saved as `flow_model_checkpoint.tar` and `flow_model.tar`.
  - A previous checkpoint can be loaded by setting `model.file_name` in `conf/config.yaml`.

- `script_model.py` converts a trained model checkpoint into a TorchScript file.
  - It loads `flow_model.tar` from the current folder.
  - It scripts and optimizes the model with `torch.jit`, then writes `scripted_model_*.pt`.

## Required Python libraries

The Python scripts in this folder import the following non-stdlib packages:

- `torch`
- `numpy`
- `hydra-core`
- `omegaconf`
- `hydra-joblib-launcher`
- `joblib`

A minimal install looks like:

```bash
pip install torch numpy hydra-core omegaconf hydra-joblib-launcher joblib
```

## Typical usage

Train the model:

```bash
python train.py
```

Export a TorchScript model:

```bash
python script_model.py
```

`script_model.py` expects the model-only weights file to be named `flow_model.tar`.
