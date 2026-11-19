# Launch the live fall-detection demo (scripts/demo_webcam.py).
#
#   make webcam                       # webcam 0 on CPU
#   make webcam-gpu                   # webcam 0 on GPU
#   make webcam CAM=1                 # another camera
#   make webcam ARGS="--no-flip --record recordings/run.mp4"
#
# Keys in the window: q / Esc quit, r reset, f toggle mirror.

CAM  ?= 0
ARGS ?=
PY_CPU := .venv-cpu/bin/python
PY_GPU := .venv-gpu/bin/python
DEMO   := scripts/demo_webcam.py

.PHONY: help webcam webcam-gpu

help:
	@sed -n '2,8s/^# \{0,1\}//p' Makefile

webcam: | $(PY_CPU)
	$(PY_CPU) $(DEMO) --source $(CAM) $(ARGS)

webcam-gpu: | $(PY_GPU)
	$(PY_GPU) $(DEMO) --gpu --source $(CAM) $(ARGS)

$(PY_CPU):
	UV_PROJECT_ENVIRONMENT=.venv-cpu uv sync --extra cpu

$(PY_GPU):
	UV_PROJECT_ENVIRONMENT=.venv-gpu uv sync --extra gpu
