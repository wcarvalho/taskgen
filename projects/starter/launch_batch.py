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

# ======================================================
# Project wide code
# ======================================================

DEFAULT_LOG_FILE="projects/starter/batch_log"
SCRIPT_FILE="projects/starter/launch_batch_helper.py"

# ======================================================
# Generic code
# ======================================================
import copy
import itertools
import multiprocessing

import torch
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.variant import make_variants, VariantLevel
from sklearn.model_selection import ParameterGrid

# ======================================================
# CHANGE below
# ======================================================
from utils.exp_launcher import run_experiments


# ======================================================
# load latext experiment from log file
# ======================================================
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default=DEFAULT_LOG_FILE)
args, unknown = parser.parse_known_args()

log_file = args.log.replace("/", ".")

from importlib import import_module
log = import_module(log_file)



# ======================================================
# Get settings + run
# ======================================================
# Either manually set the resources for the experiment:
n_cpu_core=min(log.n_cpu_core, multiprocessing.cpu_count())
n_gpu=min(log.n_gpu, torch.cuda.device_count())

ntrx = n_gpu*log.contexts_per_gpu
# make sure evenly divisible
if n_cpu_core % ntrx != 0:
    n_cpu_core = (n_cpu_core // ntrx)*ntrx

affinity_code = encode_affinity(
    n_cpu_core=n_cpu_core,
    n_gpu=n_gpu,
    contexts_per_gpu=log.contexts_per_gpu,
    # hyperthread_offset=8,  # if auto-detect doesn't work, number of CPU cores
    # n_socket=1,  # if auto-detect doesn't work, can force (or force to 1)
    # cpu_per_run=1,
    set_affinity=True,  # it can help to restrict workers to individual CPUs
)
# Or try an automatic one, but results may vary:
# affinity_code = quick_affinity_code(n_parallel=None, use_gpu=True)

def get_variant_level(search_space):
    # ======================================================
    # build up search space
    # ======================================================
    full_grid = [{}]
    keys = []
    for key, search in search_space.items():
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
    to_skip = getattr(log, 'filename_skip', [])
    value_names_file = list(filter(lambda v: not v in to_skip, value_names))

    dir_names = []
    for g in full_grid:
        dir_names.append(",".join([f"{k}={g[k]}" for k in value_names_file]))

    # ======================================================
    # create variants
    # ======================================================
    keys = sorted(keys, key=lambda x: x[1])
    values = [[g[k] for k in value_names] for g in full_grid]
    return VariantLevel(keys, values, dir_names)



# variant_levels = list()
variants = []
log_dirs = []
# ======================================================
# get Variant levels
# ======================================================
if isinstance(log.search_space, dict):
    variants, log_dirs = make_variants(get_variant_level(log.search_space))
    # variant_levels.append(get_variant_level(log.search_space))
elif isinstance(log.search_space, list):
    for search_space in log.search_space:
        _variants, _log_dirs = make_variants(get_variant_level(search_space))
        variants.extend(_variants)
        log_dirs.extend(_log_dirs)
        # variant_levels.append(get_variant_level(search_space))
else:
    raise NotImplementedError




# Between variant levels, make all combinations.
# variants, log_dirs = make_variants(*variant_levels)
for idx, variant in enumerate(variants):
    if not 'settings' in variant:
        variant['settings'] = dict()
    variant['settings']['variant_idx'] = idx


print("="*50)
print("="*50)
print(f"Running {int(len(variants)*log.runs_per_setting)} experiments")
print("="*50)
print("="*50)


run_experiments(
    script=SCRIPT_FILE,
    affinity_code=affinity_code,
    experiment_title=log.experiment_title,
    runs_per_setting=log.runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    log_dir_kwargs=dict(
        root_log_dir='data',
        time=False,
        date=True,
    )
)