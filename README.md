# Installation

```
bash setup.sh
```

# Server

```
DISPLAY=:0.X jupyter lab --port XXX --no-browser --ip 0.0.0.0
```

# Layout
* experiments: scripts for launching experiments
* models: saved parameters
* sfgen: main code
	* babyai: agent code
	* babyai_litchen: env code
	* general: losses + replay buffers + etc.
	* tools: plotting + video maker + etc.
