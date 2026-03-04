import math
import os
import sys
import subprocess

def calculate_optimal_resources(requested_threads: int, requested_workers: int, available_cores: int):
    """Rescales resources to fill the NUMA node without oversubscription."""
    # Allow some oversubscription if requested, but not more than 4 extra cores.
    usable_cores = max(1, available_cores - 1) # minus one for main thread.
    usable_cores = min(usable_cores + 4, requested_threads + requested_workers)

    if requested_threads < 0:
        requested_threads = usable_cores
    if requested_workers < 0:
        requested_workers = usable_cores

    # Defaults and safety
    req_t = max(1, requested_threads)
    req_w = max(0, requested_workers)

    # Scale both numbers to fit exactly into usable_cores
    scale = min(1.0, usable_cores / (req_t + req_w))

    actual_workers = max(0, math.floor(req_w * scale))
    actual_threads = max(1, math.ceil(req_t * scale))

    return actual_threads, actual_workers

def enforce_gpu_numa_affinity():
    """
    Locks the current process to the CPU cores attached to its assigned GPU's NUMA node.
    Returns the integer number of CPU cores bound to the process for thread pool configuration.
    """
    # Default fallback: return the number of cores the OS currently allows this process to use
    def get_default_core_count():
        if hasattr(os, 'sched_getaffinity'):
            return len(os.sched_getaffinity(0))
        return os.cpu_count() or 1

    if sys.platform != "linux":
        return get_default_core_count()

    local_rank_str = os.environ.get("LOCAL_RANK")
    if local_rank_str is None:
        return get_default_core_count()

    try:
        local_rank = int(local_rank_str)
    except ValueError:
        return get_default_core_count()

    # Map LOCAL_RANK (visible device index) to the system GPU index expected by nvidia-smi.
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        # CUDA_VISIBLE_DEVICES is a comma-separated list of physical GPU indices.
        visible_list = [d.strip() for d in cuda_visible_devices.split(",") if d.strip()]
        # Ensure local_rank is within the range of visible devices.
        if local_rank < 0 or local_rank >= len(visible_list):
            return get_default_core_count()
        try:
            physical_gpu_index = int(visible_list[local_rank])
        except ValueError:
            # Malformed CUDA_VISIBLE_DEVICES entry; fall back safely.
            return get_default_core_count()
    else:
        # No remapping; assume LOCAL_RANK matches the system GPU index.
        physical_gpu_index = local_rank

    try:
        # 1. Fetch PCI Bus ID gracefully
        try:
            pci_bus_id_raw = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=pci.bus_id", "-i", str(physical_gpu_index), "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL
            ).strip().lower()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return get_default_core_count()

        pci_bus_id = pci_bus_id_raw if ":" in pci_bus_id_raw and len(pci_bus_id_raw.split(":")[0]) >= 4 else f"0000:{pci_bus_id_raw}"

        # 2. Look up NUMA node
        numa_node_path = f"/sys/bus/pci/devices/{pci_bus_id}/numa_node"
        if not os.path.exists(numa_node_path):
            return get_default_core_count()

        with open(numa_node_path, "r") as f:
            numa_node = int(f.read().strip())

        if numa_node < 0:
            return get_default_core_count()

        # 3. Extract physical CPU cores
        cpulist_path = f"/sys/devices/system/node/node{numa_node}/cpulist"
        if not os.path.exists(cpulist_path):
            return get_default_core_count()

        with open(cpulist_path, "r") as f:
            cpulist_str = f.read().strip()

        numa_cpus = set()
        if cpulist_str:
            for part in cpulist_str.split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    numa_cpus.update(range(start, end + 1))
                else:
                    numa_cpus.add(int(part))

        if not numa_cpus:
            return get_default_core_count()

        # 4. Cgroup / Container Safety Check
        allowed_cpus = os.sched_getaffinity(0)
        target_cpus = numa_cpus.intersection(allowed_cpus)

        if not target_cpus:
            return get_default_core_count()

        # 5. Apply safe affinity and return the explicit count
        os.sched_setaffinity(0, target_cpus)
        return len(target_cpus)

    except Exception:
        # On any unhandled OS exception, revert to the safe default
        return get_default_core_count()
