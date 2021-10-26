# Install

```bash

bash projects/thor_nav/setup.sh gpu
bash projects/thor_nav/setup.sh cpu
```

# Running Experiments

**setup envionment**
```bash
# setup env
export PYTHONPATH="${PYTHONPATH}:." # otherwise doesn't recognize library
# startup x-server
xinit&
```

**run 1 experiment**
```bash
# run single experiment
python projects/thor_nav/launch_individual.py
# run single experiment on 1st gpu
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0.0 python projects/thor_nav/launch_individual.py
```

**run experiments in parallel**
```bash
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
<!-- - `nnmodules/thor_resnet_model.py` - architecture -->
- environment: `envs/thor_nav/env.py`
  - reward function: `envs/thor_nav/env.py:207`
- curriculum: `envs/thor_nav/env.py:reset():144`
- model: `nnmodules/thor_resnet_model.py`
- training algorithm: 
  -  PPO: https://github.com/astooke/rlpyt/blob/master/rlpyt/algos/pg/ppo.py
  - can copy over and import own

In order to preload parameters do the following
```python
# launch_individual.py:166
agent = BabyAIPPOAgent(
        **config[‘agent’],
        ModelCls=ThorModel,
        model_kwargs=config[‘model’],
        # CHANGE
        agent_kwargs=dict(
            initial_model_state_dict=ckpt,
            )
        )
```
