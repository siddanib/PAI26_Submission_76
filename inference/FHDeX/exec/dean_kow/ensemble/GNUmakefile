PRECISION  = FLOAT
PROFILE    = FALSE

DEBUG      = TRUE
DEBUG      = FALSE

DIM        = 2

MAX_SPEC   = 8

COMP	   = gnu

USE_MPI    = TRUE
USE_OMP    = FALSE
USE_CUDA   = FALSE

USE_PARTICLES = TRUE

Bpack   := ./Make.package
Blocs   := .

include Make.Adv

# Pytorch directories
ifeq ($(USE_CUDA),TRUE)
  PYTORCH_ROOT ?= /path/to/libtorch_cuda
else
  PYTORCH_ROOT ?= /path/to/libtorch_cpu
endif
TORCH_LIBPATH = $(PYTORCH_ROOT)/lib

ifeq ($(USE_CUDA),TRUE)
  TORCH_LIBS = -ltorch -ltorch_cpu -lc10 -lc10_cuda -lcuda
else
  TORCH_LIBS = -ltorch -ltorch_cpu -lc10
endif

INCLUDE_LOCATIONS += $(PYTORCH_ROOT)/include \
                     $(PYTORCH_ROOT)/include/torch/csrc/api/include
LIBRARY_LOCATIONS += $(TORCH_LIBPATH)

DEFINES += -D_GLIBCXX_USE_CXX11_ABI=1
ifeq ($(USE_CUDA),TRUE)
  LDFLAGS += -Xlinker "--no-as-needed,-rpath $(TORCH_LIBPATH) $(TORCH_LIBS)"
else
  LDFLAGS += -Wl,--no-as-needed,-rpath=$(TORCH_LIBPATH) $(TORCH_LIBS)
endif
