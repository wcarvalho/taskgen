# Installation

```bash
bash setup.sh
```



# Server

Launching jupyter lab:

```bash
DISPLAY=:0.X jupyter lab --port XXX --no-browser --ip 0.0.0.0
```



# Running Experiments

### Generalization with SF over Learned Factored States (sfgen)
```bash
python scripts/generate_babyai_kitchen_vocab_tasks.py # to reload vocab
export PYTHONPATH="${PYTHONPATH}:."	# otherwise doesn't recognize sfgen
python launchers/sfgen/launch_individual.py		# single experiment
python launchers/sfgen/launch_batch.py						# batched experiments in parallel
```



# Directory Logic:

### Running experiments
* `analysis`: jupyter notebooks used to analyze experiments
* `data`: where data from experiments lives
* `launchers`: files used to launch individual/batch experiments. each 
    subdirectory is assumed to belong to one project.
* `preloads`: preload data to load (e.g. vocabulary)
* `scripts`: misc. utility scripts
* `videos`: videos of agent are/should be stored here

### Adding to codebase more generally
* `agents`: python files defining classes for agents. each environment is 
    assumed to have its own agent
* `algos`: RL algorithms including helpers such as replay buffers
* `envs`: each subdirectory contains environment implementation
    * `rlpyt`: contains environment wrappers to use with `rlpyt codebase`
* `nnmodules`: pytorch `nn.Module` modules
* `utils`: misc. utility scripts (plotting, video maker, etc.)



### Get directory structure

```bash
tree -L 3 -I '_gym-minigrid|_babyai|_rlpyt|videos|*.sh|*.yaml|*sublime*|*.pyc|data|experiments|analysis'
```
