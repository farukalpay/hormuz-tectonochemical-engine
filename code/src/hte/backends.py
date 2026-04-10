from __future__ import annotations

import platform

from .types import TensorFlowProbe


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


def tensorflow_status(preference: str = "gpu") -> TensorFlowProbe:
    requested = preference.strip().lower() if preference else "gpu"
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

    physical_devices = tuple(f"{device.device_type}:{device.name}" for device in tf.config.list_physical_devices())
    visible_devices = tuple(f"{device.device_type}:{device.name}" for device in tf.config.experimental.get_visible_devices())
    gpu_devices = [device for device in tf.config.list_physical_devices("GPU")]
    metal_devices = tuple(device.name for device in gpu_devices)
    resolved_device = "/CPU:0"

    try:
        if requested in {"gpu", "metal", "auto"} and gpu_devices:
            resolved_device = "/GPU:0"
            with tf.device(resolved_device):
                matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
                tf.matmul(matrix, matrix).numpy()
            notes.append("metal-compatible GPU probe succeeded")
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

    with tf.device(resolved_device):
        matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        tf.matmul(matrix, matrix).numpy()
    if not gpu_devices:
        notes.append("no TensorFlow GPU device is visible; CPU will be used")
    elif requested in {"cpu", "force-cpu"}:
        notes.append("CPU was requested explicitly")
    elif resolved_device == "/CPU:0":
        notes.append("CPU fallback path completed successfully")

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
    probe = tensorflow_status(preference=preference)
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
    }
