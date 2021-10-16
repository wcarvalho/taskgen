#Install

```bash

bash projects/thor_nav/setup.sh gpu
bash projects/thor_nav/setup.sh cpu
```

# Running

```bash
# setup env
export PYTHONPATH="${PYTHONPATH}:." # otherwise doesn't recognize sfgen
# startup x-server
xinit&

# run stuff
python projects/thor_nav/launch_individual.py        # single experiment
python projects/thor_nav/launch_batch.py                      # batched experiments in parallel
```

# Editing Runs

- `individual_log.py` specifies latest settings to run for `launch_individual.py`
- `batch_log.py` specifies latest search to run for `launch_batch.py`
