# Install

```bash

bash projects/thor_nav/setup.sh gpu
bash projects/thor_nav/setup.sh cpu
```

# Running

```bash
# setup env
export PYTHONPATH="${PYTHONPATH}:." # otherwise doesn't recognize library
# startup x-server
xinit&


# run single experiment
python projects/thor_nav/launch_individual.py
# run single experiment on 1st gpu
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0.0 python projects/thor_nav/launch_individual.py

# run batch of experiments
python projects/thor_nav/launch_batch.py
```

# Editing Runs

- `individual_log.py` specifies latest settings to run for `launch_individual.py`
- `batch_log.py` specifies latest **parallel search** to run for `launch_batch.py`

# Killing

```bash
kill -9 $(pgrep thor)
kill -9 $(pgrep python)
```


# Main files

- `nnmodules/thor_resnet_model.py` - architecture
- `envs/thor_nav/env.py` - environment that defines task