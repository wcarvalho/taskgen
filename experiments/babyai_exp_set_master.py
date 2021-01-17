"""
Launches multiple experiment runs and organizes them on the local
compute resource.
Processor (CPU and GPU) affinities are all specified, to keep each
experiment on its own hardware without interference.  Can queue up more
experiments than fit on the machine, and they will run in order over time.  
To understand rules and settings for affinities, try using 
affinity = affinity.make_affinity(..)
OR
code = affinity.encode_affinity(..)
slot_code = affinity.prepend_run_slot(code, slot)
affinity = affinity.affinity_from_code(slot_code)
with many different inputs to encode, and see what comes out.
The results will be logged with a folder structure according to the
variant levels constructed here.
"""
import copy
import itertools
from pprint import pprint
from sklearn.model_selection import ParameterGrid

from rlpyt.utils.launching.affinity import encode_affinity, quick_affinity_code
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

import experiments.master_log as log

# Either manually set the resources for the experiment:
affinity_code = encode_affinity(
    n_cpu_core=32,
    n_gpu=4,
    # hyperthread_offset=8,  # if auto-detect doesn't work, number of CPU cores
    # n_socket=1,  # if auto-detect doesn't work, can force (or force to 1)
    # cpu_per_run=1,
    set_affinity=True,  # it can help to restrict workers to individual CPUs
)
# Or try an automatic one, but results may vary:
# affinity_code = quick_affinity_code(n_parallel=None, use_gpu=True)

variant_levels = list()
keys = []

# ======================================================
# build up search space
# ======================================================
full_grid = [{}]
for key, search in log.search_space.items():
    variables = search.keys()
    keys.extend(list(itertools.product([key], variables)))

    local_grid = []
    for g in copy.deepcopy(full_grid):
        new_search = dict(search, **{k:[v] for k,v in g.items()})
        new = list(ParameterGrid(new_search))
        local_grid.extend(new)

    full_grid = local_grid



# ======================================================
# create names of directories
# ======================================================
value_names = full_grid[0].keys()
dir_names = []
for g in full_grid:
    dir_names.append(",".join([f"{k}={g[k]}" for k in value_names]))

# ======================================================
# create variants
# ======================================================
keys = sorted(keys, key=lambda x: x[1])
values = [[g[k] for k in value_names] for g in full_grid]
variant_levels.append(VariantLevel(keys, values, dir_names))

# Between variant levels, make all combinations.
variants, log_dirs = make_variants(*variant_levels)


run_experiments(
    script="experiments/babyai_exp_set.py",
    affinity_code=affinity_code,
    experiment_title=log.experiment_title,
    runs_per_setting=log.runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
)