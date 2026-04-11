from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from .types import TensorFlowProbe

_GPU_PCI_CLASS_MARKERS = ("vga compatible controller", "3d controller", "display controller")


def _install_plan() -> list[str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and "arm" in machine:
        return [
            "/opt/homebrew/bin/python3.11 -m venv .venv-tf",
            "source .venv-tf/bin/activate",
            "pip install -r code/requirements.txt",
            "pip install -r code/requirements-tf-macos.txt",
            "pip install -e ./code[dev,tensorflow-apple]",
            ".venv-tf/bin/python code/scripts/check_tensorflow_backend.py",
        ]
    return [
        "python3 -m venv .venv-tf",
        "source .venv-tf/bin/activate",
        "pip install -r code/requirements.txt",
        "pip install -r code/requirements-tf-linux-windows.txt",
        "pip install -e ./code[dev,tensorflow]",
        ".venv-tf/bin/python code/scripts/check_tensorflow_backend.py",
    ]


def _run_output(command: list[str], timeout_seconds: float = 2.0) -> str:
    executable = command[0]
    if shutil.which(executable) is None:
        return ""
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except Exception:
        return ""
    stdout = completed.stdout.strip()
    if stdout:
        return stdout
    return completed.stderr.strip()


def _gpu_pci_devices() -> tuple[str, ...]:
    output = _run_output(["lspci", "-nn"])
    if not output:
        return ()
    devices: list[str] = []
    for line in output.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if any(marker in lowered for marker in _GPU_PCI_CLASS_MARKERS):
            devices.append(candidate)
    return tuple(devices)


def _gpu_vendors(pci_devices: tuple[str, ...]) -> tuple[str, ...]:
    vendors: list[str] = []
    for device in pci_devices:
        lowered = device.lower()
        if "nvidia" in lowered:
            vendor = "nvidia"
        elif "advanced micro devices" in lowered or "[amd/ati]" in lowered or " ati" in lowered:
            vendor = "amd"
        elif "intel" in lowered:
            vendor = "intel"
        else:
            vendor = "unknown"
        if vendor not in vendors:
            vendors.append(vendor)
    return tuple(vendors)


def accelerator_context() -> dict[str, object]:
    pci_devices = _gpu_pci_devices()
    host_gpu_vendor_hint = os.environ.get("HTE_HOST_GPU_VENDOR_HINT", "").strip().lower()
    if host_gpu_vendor_hint not in {"", "nvidia", "amd", "intel", "unknown"}:
        host_gpu_vendor_hint = ""
    tensorflow_distribution = os.environ.get("HTE_TENSORFLOW_DISTRIBUTION", "").strip().lower()
    if tensorflow_distribution not in {"", "auto", "cpu", "cuda", "rocm", "none"}:
        tensorflow_distribution = ""
    nvidia_device_nodes = sorted(path.name for path in Path("/dev").glob("nvidia*"))
    render_nodes = sorted(path.name for path in Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").exists() else []
    rocm_device_nodes_available = Path("/dev/kfd").exists() and bool(render_nodes)
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "is_container": Path("/.dockerenv").exists(),
        "gpu_pci_devices": list(pci_devices),
        "gpu_vendors": list(_gpu_vendors(pci_devices)),
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "nvidia_device_nodes": nvidia_device_nodes,
        "rocm_device_nodes_available": rocm_device_nodes_available,
        "rocm_tools_available": shutil.which("rocminfo") is not None or shutil.which("rocm-smi") is not None,
        "cuda_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", "").strip(),
        "host_gpu_vendor_hint": host_gpu_vendor_hint,
        "tensorflow_distribution": tensorflow_distribution,
        "tensorflow_built_with_cuda": False,
    }


def _append_resolution_notes(
    notes: list[str],
    *,
    requested: str,
    gpu_devices: list[object],
    resolved_device: str,
    context: dict[str, object],
    built_with_cuda: bool,
) -> None:
    if requested in {"cpu", "force-cpu"}:
        notes.append("CPU was requested explicitly")
        return

    if gpu_devices and resolved_device == "/CPU:0":
        notes.append("CPU fallback path completed successfully")
        return

    if gpu_devices:
        return

    notes.append("no TensorFlow GPU device is visible; CPU will be used")
    nvidia_nodes = context.get("nvidia_device_nodes", [])
    if built_with_cuda and not context.get("nvidia_smi_available") and not nvidia_nodes:
        notes.append("TensorFlow CUDA build is installed, but no NVIDIA driver/device is visible in this runtime.")
    vendors = context.get("gpu_vendors", [])
    host_vendor_hint = str(context.get("host_gpu_vendor_hint") or "").strip().lower()
    amd_hint = (isinstance(vendors, list) and "amd" in vendors) or host_vendor_hint == "amd"
    if amd_hint and built_with_cuda:
        notes.append("TensorFlow runtime targets CUDA/NVIDIA; AMD GPUs require a ROCm TensorFlow runtime.")
    if amd_hint and not context.get("rocm_device_nodes_available"):
        notes.append("AMD GPU is present in PCI inventory, but ROCm device nodes are unavailable (/dev/kfd and /dev/dri/renderD*).")
    if amd_hint and context.get("rocm_device_nodes_available") and not gpu_devices:
        notes.append("ROCm device nodes are present, but TensorFlow still reports no GPU. Check ROCm image compatibility and GPU architecture support.")
    if context.get("is_container") and not nvidia_nodes and not context.get("rocm_device_nodes_available"):
        notes.append("Container has no GPU device mounts; use --gpus all (NVIDIA) or ROCm device mounts (AMD).")


def _configure_tensorflow_runtime(tf) -> tuple[list[object], list[str]]:
    runtime_notes: list[str] = []
    gpu_devices = list(tf.config.list_physical_devices("GPU"))

    try:
        tf.config.optimizer.set_jit(False)
        runtime_notes.append("TensorFlow XLA JIT is disabled for training stability.")
    except Exception:
        pass

    for device in gpu_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except Exception:
            continue
    if gpu_devices:
        runtime_notes.append("TensorFlow GPU memory growth is enabled.")

    return gpu_devices, runtime_notes


def tensorflow_status(
    preference: str = "gpu",
    *,
    context: dict[str, object] | None = None,
    active_probe: bool = True,
) -> TensorFlowProbe:
    requested = preference.strip().lower() if preference else "gpu"
    accelerator = accelerator_context() if context is None else context
    notes: list[str] = []
    try:
        import tensorflow as tf
    except ImportError:
        notes.append("tensorflow is not installed in the active environment")
        notes.extend(_install_plan())
        return TensorFlowProbe(
            available=False,
            version=None,
            requested_preference=requested,
            resolved_device="/CPU:0",
            physical_devices=(),
            visible_devices=(),
            metal_devices=(),
            probe_operation="matmul(2x2)",
            probe_success=False,
            notes=tuple(notes),
        )

    gpu_devices, runtime_notes = _configure_tensorflow_runtime(tf)
    notes.extend(runtime_notes)
    built_with_cuda = bool(getattr(tf.test, "is_built_with_cuda", lambda: False)())
    accelerator["tensorflow_built_with_cuda"] = built_with_cuda
    physical_devices = tuple(f"{device.device_type}:{device.name}" for device in tf.config.list_physical_devices())
    visible_devices = tuple(f"{device.device_type}:{device.name}" for device in tf.config.experimental.get_visible_devices())
    metal_devices = tuple(device.name for device in gpu_devices)
    resolved_device = "/CPU:0"

    if requested in {"gpu", "metal", "auto"} and gpu_devices:
        resolved_device = "/GPU:0"
        if not active_probe:
            notes.append("TensorFlow GPU device is visible; active probe was skipped.")
            return TensorFlowProbe(
                available=True,
                version=tf.__version__,
                requested_preference=requested,
                resolved_device=resolved_device,
                physical_devices=physical_devices,
                visible_devices=visible_devices,
                metal_devices=metal_devices,
                probe_operation="visibility-only",
                probe_success=False,
                notes=tuple(notes),
            )

        try:
            with tf.device(resolved_device):
                matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
                tf.matmul(matrix, matrix).numpy()
            notes.append("TensorFlow GPU probe succeeded")
            return TensorFlowProbe(
                available=True,
                version=tf.__version__,
                requested_preference=requested,
                resolved_device=resolved_device,
                physical_devices=physical_devices,
                visible_devices=visible_devices,
                metal_devices=metal_devices,
                probe_operation="matmul(2x2)",
                probe_success=True,
                notes=tuple(notes),
            )
        except Exception as exc:  # pragma: no cover - depends on host-specific Metal behavior
            notes.append(f"GPU probe failed and training will fall back to CPU: {exc}")
            resolved_device = "/CPU:0"

    if not active_probe:
        _append_resolution_notes(
            notes,
            requested=requested,
            gpu_devices=gpu_devices,
            resolved_device=resolved_device,
            context=accelerator,
            built_with_cuda=built_with_cuda,
        )
        return TensorFlowProbe(
            available=True,
            version=tf.__version__,
            requested_preference=requested,
            resolved_device=resolved_device,
            physical_devices=physical_devices,
            visible_devices=visible_devices,
            metal_devices=metal_devices,
            probe_operation="visibility-only",
            probe_success=False,
            notes=tuple(notes),
        )

    with tf.device(resolved_device):
        matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        tf.matmul(matrix, matrix).numpy()
    _append_resolution_notes(
        notes,
        requested=requested,
        gpu_devices=gpu_devices,
        resolved_device=resolved_device,
        context=accelerator,
        built_with_cuda=built_with_cuda,
    )

    return TensorFlowProbe(
        available=True,
        version=tf.__version__,
        requested_preference=requested,
        resolved_device=resolved_device,
        physical_devices=physical_devices,
        visible_devices=visible_devices,
        metal_devices=metal_devices,
        probe_operation="matmul(2x2)",
        probe_success=True,
        notes=tuple(notes),
    )


def backend_payload(preference: str = "gpu") -> dict[str, object]:
    context = accelerator_context()
    probe = tensorflow_status(preference=preference, context=context)
    return {
        "available": probe.available,
        "version": probe.version,
        "requested_preference": probe.requested_preference,
        "resolved_device": probe.resolved_device,
        "physical_devices": list(probe.physical_devices),
        "visible_devices": list(probe.visible_devices),
        "metal_devices": list(probe.metal_devices),
        "probe_operation": probe.probe_operation,
        "probe_success": probe.probe_success,
        "notes": list(probe.notes),
        "built_with_cuda": bool(context.get("tensorflow_built_with_cuda")),
        "accelerator_context": context,
    }
