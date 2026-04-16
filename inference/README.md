# Inference Directory

This directory contains the inference-oriented setup for the `FHDeX` Dean-Kawasaki / random-walker simulations, along with example input files for Gaussian, Markovian ML, and non-Markovian ML flux modes.

## Contents

- `FHDeX/`: source tree and build files for the simulation code
- `inputs_Dean_Kawasaki`: example input file using `flux_mode = "gaussian"`
- `inputs_Markovian_ML`: example input file for Markovian ML inference
- `inputs_Non_Markovian_ML`: example input file for non-Markovian ML inference

## Relevant Subfolders

- `FHDeX/exec/dean_kow/ensemble/`: executable-specific build configuration
  - `Make.Adv`: AMReX configuration and source inclusion
  - `GNUmakefile`: main compilation options, including PyTorch / libtorch settings
- `FHDeX/src_deankow/ensemble/`: Dean-Kawasaki application source
- `FHDeX/src_common/`: shared utilities and common simulation routines
- `FHDeX/src_rng/`: random number generation support

## External Dependencies

This code depends on:

- MPI
- A C++ compiler supported by the AMReX build configuration
- [AMReX](https://github.com/AMReX-Codes/amrex)
- [libtorch](https://pytorch.org/get-started/locally/)

The vendored `FHDeX/` subtree also has its own upstream repository reference:

- <https://github.com/AMReX-FHD/FHDeX.git>

## Build Setup

Build from:

- `FHDeX/exec/dean_kow/ensemble/`

Before compiling, update the following paths:

1. In [FHDeX/exec/dean_kow/ensemble/Make.Adv](./FHDeX/exec/dean_kow/ensemble/Make.Adv), set `AMREX_HOME` to your AMReX installation path.

   Current placeholder:

   ```make
   AMREX_HOME ?= /path/to/amrex
   ```

2. Download `libtorch` and set `PYTORCH_ROOT` in [FHDeX/exec/dean_kow/ensemble/GNUmakefile](./FHDeX/exec/dean_kow/ensemble/GNUmakefile) to the extracted libtorch directory.

   Current placeholders:

   ```make
   ifeq ($(USE_CUDA),TRUE)
     PYTORCH_ROOT ?= /path/to/libtorch_cuda
   else
     PYTORCH_ROOT ?= /path/to/libtorch_cpu
   endif
   ```

   The `libtorch` build must match the setting of `USE_CUDA` in `GNUmakefile`:

   - `USE_CUDA = FALSE`: use a CPU libtorch distribution
   - `USE_CUDA = TRUE`: use a CUDA-enabled libtorch distribution

After those paths are updated, build from:

```bash
cd FHDeX/exec/dean_kow/ensemble
make
```

For a parallel build, run:

```bash
cd FHDeX/exec/dean_kow/ensemble
make -j 4
```

The executable is produced in the build directory and should be used from there unless you move it elsewhere deliberately.

## Input Files

The example input files at the top level are:

- `inputs_Dean_Kawasaki`: Gaussian flux baseline example
- `inputs_Markovian_ML`: ML-driven run with `ml_history_len = 0`
- `inputs_Non_Markovian_ML`: ML-driven run with `ml_history_len = 10`

For the ML input files, the scripted Torch model must be provided through the `ml_model_file` entry. In both example files, it currently points to:

```text
ml_model_file = "./scripted_model.pt"
```

Make sure the scripted model file exists at that location, or replace it with the correct path before running:

- [inputs_Markovian_ML](./inputs_Markovian_ML)
- [inputs_Non_Markovian_ML](./inputs_Non_Markovian_ML)

The model file should be a TorchScript / scripted model compatible with the libtorch C++ runtime.

## Usage

Populate this section with the executable name produced by your local build. Example commands:

```bash
mpirun -np 4 executable inputs_Dean_Kawasaki
mpirun -np 4 executable inputs_Markovian_ML
mpirun -np 4 executable inputs_Non_Markovian_ML
```

Run from the directory that contains the executable, or adjust file paths in the input files accordingly. This is especially important for relative paths such as:

```text
ml_model_file = "./scripted_model.pt"
```

## Output

The simulation writes AMReX plotfiles using the prefix specified by `amr.plot_file` in the input file. These plotfiles contain at least the following variables:

- `phi0`: SPDE number-density solution
- `phi1`: number-density solution from random-walker particles

Checkpoint files are also written according to `amr.chk_file` and `amr.chk_int`.

Plotfiles and checkpoint files are written to the run directory.

## Troubleshooting

- If compilation fails while including AMReX make infrastructure, re-check `AMREX_HOME` in `Make.Adv`.
- If compilation or linking fails with missing Torch headers or libraries, re-check `PYTORCH_ROOT` in `GNUmakefile`.
- If an ML run fails at startup, verify that `ml_model_file` points to an existing TorchScript model and that the path is correct relative to the run directory.

## Notes

- `inputs_Dean_Kawasaki` currently uses `flux_mode = "gaussian"`.
- The ML input files use `flux_mode = "ml"` and require a valid scripted model path before execution.
- Additional details about the embedded codebase are available in [FHDeX/README.md](./FHDeX/README.md).
