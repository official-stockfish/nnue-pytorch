import os
import sys
import subprocess
import math

def _parse_cpu_list(cpu_str: str) -> set:
    """Parses a sysfs cpu list string (e.g., '0,2,4-7') into a set of integers."""
    cpus = set()
    if not cpu_str:
        return cpus
    for part in cpu_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return cpus

def enforce_gpu_numa_affinity():
    """
    Locks the process to CPU cores attached to its GPU's NUMA node.
    Partitions cores evenly if multiple GPUs share the node, keeping SMT siblings grouped.
    Returns the integer number of CPU cores bound to the process.
    """
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

    # 1. Determine Visible GPUs
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        visible_list = [d.strip() for d in cuda_visible.split(",") if d.strip()]
    else:
        try:
            gpu_indices_str = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL
            ).strip()
            visible_list = gpu_indices_str.split("\n") if gpu_indices_str else []
        except Exception:
            return get_default_core_count()

    if local_rank < 0 or local_rank >= len(visible_list):
        return get_default_core_count()

    # 2. Map all visible GPUs to their NUMA nodes via PCI bus
    gpu_numa_map = []
    try:
        pci_bus_ids_raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL
        ).strip().lower().split("\n")

        pci_map = {}
        for line in pci_bus_ids_raw:
            if "," in line:
                idx_str, pci_str = line.split(",", 1)
                pci_map[idx_str.strip()] = pci_str.strip()

        for lr, phys_idx_str in enumerate(visible_list):
            pci_bus_id_raw = pci_map.get(phys_idx_str, "")
            if not pci_bus_id_raw:
                gpu_numa_map.append((lr, -1))
                continue

            pci_bus_id = pci_bus_id_raw if ":" in pci_bus_id_raw and len(pci_bus_id_raw.split(":")[0]) >= 4 else f"0000:{pci_bus_id_raw}"
            numa_node_path = f"/sys/bus/pci/devices/{pci_bus_id}/numa_node"

            numa_node = -1
            if os.path.exists(numa_node_path):
                with open(numa_node_path, "r") as f:
                    numa_node = int(f.read().strip())
            gpu_numa_map.append((lr, numa_node))

    except Exception:
        return get_default_core_count()

    # 3. Identify peers sharing the same NUMA node
    my_numa_node = next((node for lr, node in gpu_numa_map if lr == local_rank), -1)
    if my_numa_node < 0:
        return get_default_core_count()

    peers_on_node = sorted([lr for lr, node in gpu_numa_map if node == my_numa_node])
    my_node_index = peers_on_node.index(local_rank)
    total_peers = len(peers_on_node)

    # 4. Extract target CPU cores based on NUMA and OS cgroup limits
    cpulist_path = f"/sys/devices/system/node/node{my_numa_node}/cpulist"
    if not os.path.exists(cpulist_path):
        return get_default_core_count()

    with open(cpulist_path, "r") as f:
        numa_cpus = _parse_cpu_list(f.read().strip())

    allowed_cpus = os.sched_getaffinity(0)
    target_cpus = numa_cpus.intersection(allowed_cpus)

    if not target_cpus:
        return get_default_core_count()

    # 5. Group by SMT Siblings (Hyperthreading)
    physical_cores = set()
    for cpu in target_cpus:
        # Check modern and legacy sysfs paths for sibling lists
        sibling_paths = [
            f"/sys/devices/system/cpu/cpu{cpu}/topology/core_cpus_list",
            f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        ]

        siblings = {cpu}
        for path in sibling_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        parsed_siblings = _parse_cpu_list(f.read().strip())
                        # Keep only siblings that are currently available to us
                        valid_siblings = parsed_siblings.intersection(target_cpus)
                        if valid_siblings:
                            siblings = valid_siblings
                    break
                except Exception:
                    pass

        # Store as a sorted tuple so it can be hashed and deduplicated
        physical_cores.add(tuple(sorted(siblings)))

    # Sort the unique physical cores deterministically
    unique_cores = sorted(list(physical_cores))

    # 6. Partition the physical core groups among peers
    chunk_size = len(unique_cores) // total_peers
    remainder = len(unique_cores) % total_peers

    start_idx = my_node_index * chunk_size + min(my_node_index, remainder)
    end_idx = start_idx + chunk_size + (1 if my_node_index < remainder else 0)

    my_core_chunks = unique_cores[start_idx:end_idx]

    # Flatten the assigned chunks back into a flat list of CPUs
    my_final_cpus = [cpu for chunk in my_core_chunks for cpu in chunk]

    if not my_final_cpus:
        return get_default_core_count()

    try:
        os.sched_setaffinity(0, my_final_cpus)
        return len(my_final_cpus)
    except Exception:
        return get_default_core_count()

def setup_environment(requested_threads: int = -1, requested_workers: int = 0):
    """
    Applies OS constraints before heavy library loading.
    """
    os.environ["PYTHONUNBUFFERED"] = "1"
    available_cores = enforce_gpu_numa_affinity()

    usable_cores = max(1, available_cores - 1)
    if requested_threads < 0:
        requested_threads = usable_cores

    req_t = max(1, requested_threads)
    req_w = max(0, requested_workers)
    usable_cores = min(int(usable_cores * 1.1), req_t + req_w)
    scale = min(1.0, usable_cores / (req_t + req_w))

    actual_threads = max(1, math.ceil(req_t * scale))
    actual_workers = max(0, math.floor(req_w * scale))

    os.environ["OMP_NUM_THREADS"] = str(actual_threads)
    os.environ["MKL_NUM_THREADS"] = str(actual_threads)

    return actual_threads, actual_workers
