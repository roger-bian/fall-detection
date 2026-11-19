"""Process-level ONNX Runtime setup. Import this before anything imports onnxruntime.

onnxruntime-gpu cannot find the CUDA 12 / cuDNN 9 shared objects on its own on this
box. When the CUDA EP fails to load it silently falls back to CPUExecutionProvider,
while get_available_providers() still lists CUDA, so nothing looks wrong until you
measure throughput (shoplifting-detection docs/HARDWARE.md, 2026-08-06).

The libraries come from the nvidia-* wheels pulled in by onnxruntime-gpu[cuda,cudnn];
onnxruntime.preload_dlls() loads them into the process before a session is created.
"""

from __future__ import annotations

_done = False


def preload_cuda_libs() -> None:
    """Idempotent. Safe without a GPU: require_cuda_ep() is what decides."""
    global _done
    if _done:
        return
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    _done = True


def ort_sessions(obj, _depth: int = 0):
    """Yield every InferenceSession reachable from an rtmlib model object."""
    import onnxruntime as ort

    if _depth > 3 or not hasattr(obj, "__dict__"):
        return
    for v in vars(obj).values():
        if isinstance(v, ort.InferenceSession):
            yield v
        else:
            yield from ort_sessions(v, _depth + 1)


def require_cuda_ep(model) -> None:
    """Raise if any session in `model` quietly landed on CPU."""
    sessions = list(ort_sessions(model))
    bad = [s.get_providers() for s in sessions
           if "CUDAExecutionProvider" not in s.get_providers()]
    if not sessions or bad:
        raise RuntimeError(f"onnxruntime session(s) not on CUDA: {bad or 'no sessions found'}")


if __name__ == "__main__":
    preload_cuda_libs()
    import onnxruntime as ort

    print("onnxruntime :", ort.__version__)
    print("providers   :", ort.get_available_providers())
