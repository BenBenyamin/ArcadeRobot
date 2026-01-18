# Robot Arm Plays an Arcade Game


[![Watch the video](https://img.youtube.com/vi/FJ-UCm9jRK4/maxresdefault.jpg)](https://www.youtube.com/watch?v=FJ-UCm9jRK4)

## Project Overview
This repository contains the spring‑quarter achievements exploring how a robot arm can learn to play an arcade game (Pong) under delayed and sparse reward conditions, using reinforcement learning.

The core idea is to investigate algorithms that handle delayed feedback and sparse rewards in a physical setup. A custom interface delays state/action observations. The agent is trained end‑to‑end despite observation/action latency.
The robot used is the [Stretch3 from Hello Robotic](https://hello-robot.com/stretch-3-product).
For more details, check out the writeup [here](https://benbenyamin.github.io/ArcadeBot/). You could also check the [`/dev` branch](https://github.com/BenBenyamin/ArcadeRobot/tree/dev), which has the whole story.

**Note**: This project uses Python 3.12.

## Training the Agent

All training code lives under the `train/` directory and is designed to run locally.

### Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

Install PyTorch (CUDA 12.4 build), then the remaining requirements:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```
---

### Train

To train using the custom inertia and latency wrappers:

```bash
cd train
python train.py
```

This script:

* Creates Atari environments via **Gymnasium + ALE**
* Applies custom wrappers from `inertia_warpper.py`
* Supports PPO, A2C, DQN, SAC, QRDQN, TRPO, and RecurrentPPO
* Logs latency and training artifacts to `logs/`

---

### Train Using RLZoo Config Files (Optional)

You can also train using **RL Zoo–style YAML configs** located in `train/rlzoo_config/`.

Example:

```bash
cd train
python train_rlzoo.py
```

Available configs:

* `a2c.yml`
* `dqn.yml`
* `ppo.yml`
* `qrdqn.yml`
* `recurrentppo.yml`

This mode is useful for:

* Rapid hyperparameter sweeps
* Reproducing standardized SB3 experiments
* Comparing against baseline RL Zoo settings

---

## Onboard

The onboard code is intended to run on the robot (does not include training logic).

### Copy Code to the Robot

For example you can use:

```bash
scp -r onboard user@robot:/path/to/project/
```

---

### Create a Virtual Environment (On Robot)

On the robot:

```bash
cd onboard
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Source and Run

```bash
source .venv/bin/activate
python main.py
```

`main.py`:

* Loads a trained PPO policy (`.zip`)
* Handles real-time control
* Applies action mapping and latency compensation
* Uses icons from `onboard/ICONS/` for UI feedback

---

## Project Structure

```text
.
├── logs/                      # Training logs and latency traces
│   └── latency_log_*.csv
│
├── onboard/                   # Code deployed to the robot
│   ├── control.py             # Low-level control logic
│   ├── game.py                # Environment / interaction loop
│   ├── main.py                # Entry point (run this onboard)
│   ├── ICONS/                 # UI icons for actions
│   ├── PPO_stochastic_*.zip   # Trained policy
│   └── requirements.txt       # Minimal onboard dependencies
│
├── train/                     # Training code (local)
│   ├── inertia_warpper.py     # Custom inertia & delay wrappers
|   ├── latencies.pkl          # My latency measurements (sec)
│   ├── latency_sampler.py     # Latency modeling
│   ├── train.py               # Main training script
│   ├── train_rlzoo.py         # RL Zoo–style training
│   ├── utils.py               # Action maps & helpers
│   └── rlzoo_config/          # YAML configs for algorithms
│
├── requirements.txt           # Full training dependencies
└── README.md
```