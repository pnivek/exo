import os
import platform
import socket
import sys
from pathlib import Path
from subprocess import CalledProcessError

import psutil
from anyio import run_process

from exo.shared.types.profiling import InterfaceType, NetworkInterfaceInfo


def get_os_version() -> str:
    """Return the OS version string for this node.

    On macOS this is the macOS version (e.g. ``"15.3"``).
    On other platforms it falls back to the platform name (e.g. ``"Linux"``).
    """
    if sys.platform == "darwin":
        version = platform.mac_ver()[0]
        return version if version else "Unknown"
    return platform.system() or "Unknown"


async def get_os_build_version() -> str:
    """Return the macOS build version string (e.g. ``"24D5055b"``).

    On non-macOS platforms, returns ``"Unknown"``.
    """
    if sys.platform != "darwin":
        return "Unknown"

    try:
        process = await run_process(["sw_vers", "-buildVersion"])
    except CalledProcessError:
        return "Unknown"

    return process.stdout.decode("utf-8", errors="replace").strip() or "Unknown"


async def get_friendly_name() -> str:
    """
    Asynchronously gets the 'Computer Name' (friendly name) of a Mac.
    e.g., "John's MacBook Pro"
    Returns the name as a string, or None if an error occurs or not on macOS.
    """
    hostname = socket.gethostname()

    if sys.platform != "darwin":
        return hostname

    try:
        process = await run_process(["scutil", "--get", "ComputerName"])
    except CalledProcessError:
        return hostname

    return process.stdout.decode("utf-8", errors="replace").strip() or hostname


async def _get_interface_types_from_networksetup() -> dict[str, InterfaceType]:
    """Parse networksetup -listallhardwareports to get interface types."""
    if sys.platform != "darwin":
        return {}

    try:
        result = await run_process(["networksetup", "-listallhardwareports"])
    except CalledProcessError:
        return {}

    types: dict[str, InterfaceType] = {}
    current_type: InterfaceType = "unknown"

    for line in result.stdout.decode().splitlines():
        if line.startswith("Hardware Port:"):
            port_name = line.split(":", 1)[1].strip()
            if "Wi-Fi" in port_name:
                current_type = "wifi"
            elif "Ethernet" in port_name or "LAN" in port_name:
                current_type = "ethernet"
            elif port_name.startswith("Thunderbolt"):
                current_type = "thunderbolt"
            else:
                current_type = "unknown"
        elif line.startswith("Device:"):
            device = line.split(":", 1)[1].strip()
            # enX is ethernet adapters or thunderbolt - these must be deprioritised
            if device.startswith("en") and device not in ["en0", "en1"]:
                current_type = "maybe_ethernet"
            types[device] = current_type

    return types


def _get_linux_rdma_netdev_map() -> dict[str, str]:
    """Map network interface names to RDMA device names on Linux.

    Reads /sys/class/infiniband/*/ports/1/gid_attrs/ndevs/0 to find
    which netdev each RDMA device is bound to.
    Returns {netdev_name: rdma_device_name}.
    """
    rdma_map: dict[str, str] = {}
    infiniband_path = Path("/sys/class/infiniband")
    if not infiniband_path.exists():
        return rdma_map
    for dev_dir in infiniband_path.iterdir():
        ndev_path = dev_dir / "ports" / "1" / "gid_attrs" / "ndevs" / "0"
        state_path = dev_dir / "ports" / "1" / "state"
        if not ndev_path.exists():
            continue
        # Only include active RDMA devices
        try:
            state = state_path.read_text().strip()
            if "ACTIVE" not in state:
                continue
            netdev = ndev_path.read_text().strip()
            rdma_map[netdev] = dev_dir.name
        except OSError:
            continue
    return rdma_map


async def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information.
    On macOS, parses 'networksetup -listallhardwareports' for interface types.
    On Linux, detects RDMA devices via /sys/class/infiniband/.
    Returns a list of NetworkInterfaceInfo objects.
    """
    interfaces_info: list[NetworkInterfaceInfo] = []
    interface_types = await _get_interface_types_from_networksetup()
    rdma_map: dict[str, str] = (
        _get_linux_rdma_netdev_map() if sys.platform == "linux" else {}
    )

    for iface, services in psutil.net_if_addrs().items():
        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    interfaces_info.append(
                        NetworkInterfaceInfo(
                            name=iface,
                            ip_address=service.address,
                            interface_type=interface_types.get(iface, "unknown"),
                            rdma_device_name=rdma_map.get(iface),
                        )
                    )
                case _:
                    pass

    return interfaces_info


async def _detect_nvidia_gpu() -> str | None:
    """Try to detect NVIDIA GPU model via nvidia-smi."""
    try:
        process = await run_process(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"]
        )
        gpu_name = process.stdout.decode().strip().split("\n")[0]
        if gpu_name:
            return (
                gpu_name
                if gpu_name.upper().startswith("NVIDIA")
                else f"NVIDIA {gpu_name}"
            )
    except (CalledProcessError, FileNotFoundError):
        pass
    return None


async def get_model_and_chip() -> tuple[str, str]:
    """Get Mac system information using system_profiler."""
    model = "Unknown Model"
    chip = "Unknown Chip"

    if sys.platform != "darwin":
        model = os.environ.get("EXO_DEVICE_MODEL", model)
        chip = os.environ.get("EXO_DEVICE_CHIP", chip)
        if chip == "Unknown Chip":
            chip = await _detect_nvidia_gpu() or chip
        return (model, chip)

    try:
        process = await run_process(
            [
                "system_profiler",
                "SPHardwareDataType",
            ]
        )
    except CalledProcessError:
        return (model, chip)

    # less interested in errors here because this value should be hard coded
    output = process.stdout.decode().strip()

    model_line = next(
        (line for line in output.split("\n") if "Model Name" in line), None
    )
    model = model_line.split(": ")[1] if model_line else "Unknown Model"

    chip_line = next((line for line in output.split("\n") if "Chip" in line), None)
    chip = chip_line.split(": ")[1] if chip_line else "Unknown Chip"

    return (model, chip)
