# Project Overview

This repository is organized into two main folders:

- `training/`: Python code for training the model and exporting it as a scripted model for downstream use.
- `inference/`: C++ code for running inference within the simulation workflow using the exported model.

## Directory Summary

### `training/`

The [`training/`](./training) folder contains the Python-based training utilities for the flow-based model used in this project. It includes scripts for:

- training the model
- exporting the trained model to TorchScript

See [`training/README.md`](./training/README.md) for details.

### `inference/`

The [`inference/`](./inference) folder contains the C++-based inference and simulation setup. It includes the `FHDeX` source tree, build configuration, and example input files for Gaussian and ML-driven runs.

See [`inference/README.md`](./inference/README.md) for details.
