"""Initialize PyTorch on one CPU, then restore affinity before running a script."""
import os
from pathlib import Path
import runpy
import sys

if len(sys.argv) < 2:
    raise SystemExit("Usage: torch_entry.py SCRIPT [ARGS...]")

affinity = os.sched_getaffinity(0)
physical_cores = set()
for cpu in affinity:
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    physical_cores.add(((topology / "physical_package_id").read_text().strip(),
                        (topology / "core_id").read_text().strip()))
thread_setting = os.environ.get("MKL_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
thread_count = int(thread_setting.split(",")[0]) if thread_setting else len(physical_cores)
os.sched_setaffinity(0, {min(affinity)})
try:
    import torch
finally:
    # Native threads created during import inherit the temporary mask as well.
    for entry in Path("/proc/self/task").iterdir():
        try:
            os.sched_setaffinity(int(entry.name), affinity)
        except ProcessLookupError:
            pass

torch.set_num_threads(thread_count)
target = Path(sys.argv[1]).resolve()
sys.argv = [str(target), *sys.argv[2:]]
sys.path.insert(0, str(target.parent))
runpy.run_path(str(target), run_name="__main__")
