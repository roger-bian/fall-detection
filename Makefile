# Set up the environments and launch the live fall-detection demo (scripts/demo_webcam.py).
#
#   make install                      # create/update the CPU venv (.venv-cpu)
#   make install-gpu                  # create/update the GPU venv (.venv-gpu, CUDA ~3.3 GB)
#   make webcam                       # webcam 0 on CPU
#   make webcam-gpu                   # webcam 0 on GPU
#   make webcam 1                     # another camera (same as CAM=1)
#   make webcam ARGS="--fullscreen"   # fill the monitor (q / Esc to quit)
#   make webcam ARGS="--no-flip --record recordings/run.mp4"
#
# Keys in the window: q / Esc quit, r reset, f toggle mirror.

# `make webcam 1`: take a numeric goal after webcam / webcam-gpu as the camera index.
CAM_ARG := $(filter-out webcam webcam-gpu,$(filter webcam webcam-gpu $(shell seq 0 99),$(MAKECMDGOALS)))
ifneq ($(and $(filter webcam webcam-gpu,$(MAKECMDGOALS)),$(CAM_ARG)),)
CAM := $(firstword $(CAM_ARG))
$(CAM_ARG):
	@:
endif
CAM  ?= 0
ARGS ?=
PY_CPU := .venv-cpu/bin/python
PY_GPU := .venv-gpu/bin/python
# Downloads the default RTMO-m weights (~80 MB, cached in ~/.cache/rtmlib) at install
# time, so the first `make webcam` does not stall on it.
FETCH  := -c "import sys; sys.path.insert(0, 'src'); from falldet.pose import make_backend; make_backend('rtmo', 'cpu')"
DEMO   := scripts/demo_webcam.py

.PHONY: help install install-gpu webcam webcam-gpu

help:
	@sed -n '3,11s/^# \{0,1\}//p' Makefile

install:
	UV_PROJECT_ENVIRONMENT=.venv-cpu uv sync --extra cpu
	$(PY_CPU) -m falldet.runtime
	$(PY_CPU) $(FETCH)

install-gpu:
	UV_PROJECT_ENVIRONMENT=.venv-gpu uv sync --extra gpu
	$(PY_GPU) -m falldet.runtime
	$(PY_GPU) $(FETCH)

webcam: | $(PY_CPU)
	$(PY_CPU) $(DEMO) --source $(CAM) $(ARGS)

webcam-gpu: | $(PY_GPU)
	$(PY_GPU) $(DEMO) --gpu --source $(CAM) $(ARGS)

$(PY_CPU):
	UV_PROJECT_ENVIRONMENT=.venv-cpu uv sync --extra cpu

$(PY_GPU):
	UV_PROJECT_ENVIRONMENT=.venv-gpu uv sync --extra gpu
