
"""
Helper used to run batch experiments. Don't run this file by itself.
"""
import sys

# ======================================================
# rlpyt
# ======================================================
from rlpyt.utils.launching.variant import load_variant
from rlpyt.utils.logging import logger

# ======================================================
# Our modules
# ======================================================
from utils import get_run_name
from utils import update_config
from experiments.individual import train, load_config

def build_and_train(slot_affinity_code, log_dir, run_ID):
    variant = load_variant(log_dir)

    global config
    # ======================================================
    # load configs
    # ======================================================
    if 'settings' in variant:
        settings = variant.get('settings', dict())
        variant_idx = settings.get("variant_idx", 0)
    else:
        raise RuntimeError("settings required to get variant index")

    # ======================================================
    # first configs: env --> agent --> update with variant
    # ======================================================
    config = load_config(settings)
    config = update_config(config, variant)

    affinity = affinity_from_code(slot_affinity_code)

    if "cuda_idx" in affinity:
        gpu=True
    else:
        gpu=False

    logger.set_snapshot_gap(5e5)

    experiment_title = get_run_name(log_dir)

    train(config, affinity, log_dir, run_ID,
        name=f"{experiment_title}_var{variant_idx}",
        gpu=gpu, wandb=False, skip_launched=True)


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
