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
export PYTHONPATH="${PYTHONPATH}:."	# otherwise doesn't recognize sfgen
python launchers/sfgen/launch_individual.py		# single experiment
python launchers/sfgen/launch_batch.py						# batched experiments in parallel
```



# Directory Logic:

* `agents`: python files defining classes for agents. each environment is 
    assumed to have its own agent
* `algos`: reinforcement learning algorithms including helpers such as replay 
    buffers
* `launchers`: files used to launch individual/batch experiments. each subdirectory is assumed to belong to one project.
* `environments`: each subdirectory contains environment implementation
    * `rlpyt`: contains environment wrappers to use with `rlpyt codebase`
* `nnmodules`: pytorch `nn.Module` modules
* `utils`: misc. utility scripts

# File Layouts
```
- analysis: jupyter lab analysis related stuff
- experiments: scripts for launching experiments
- models: saved parameters
- scripts: misc. useful scripts
- sfgen: main code
    - babyai: agent code
        - configs: config files
    - babyai_kitchen: "kitchen" env code
    - general: losses + replay buffers + etc.
    - tools: plotting + video maker + etc.
- videos: generated videos
```


### Get directory structure

```bash
tree -L 3 -I '_gym-minigrid|_babyai|_rlpyt|videos|*.sh|*.yaml|*sublime*|*.pyc|data|experiments|analysis'
```
