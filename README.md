# Robot Arm Plays an Arcade Game

> **Work in Progress**

Please see `spring-report/report.md` for more detailed breakdown of the project status.

## Project Overview
This repository contains the spring‑quarter achievements exploring how a robot arm can learn to play an arcade game (Pong) under delayed and sparse reward conditions, using reinforcement learning.

The core idea is to investigate algorithms that handle delayed feedback and sparse rewards in a physical setup. A custom interface delays state/action observations. The agent is trained end‑to‑end despite observation/action latency.
The robot used is the [Stretch3 from Hello Robotics](https://hello-robot.com/stretch-3-product).

## Outcomes & Deliverables
- A set of Jupyter notebooks demonstrating **CNN‑based RL**, **computer‑vision control**, and a **VAE** for state representation.  
- Latency‑logging and analysis scripts.  
- A robot control script to validate real‑world performance under induced delays.  
- A written spring report (`spring-report/report.md`) summarizing results and insights.

## Prerequisites & Installation
1. **Python** ≥ 3.8  
2. Clone the repo:
   ```bash
   git clone git@github.com:BenBenyamin/ArcadeRobot.git
   cd ArcadeRobot
   ```
3. Install dependencies:
   ```bash
   pip install torch torchvision gymnasium stable-baselines3 opencv-python matplotlib pandas jupyterlab ale_py
   ```

## How to Run
1. **Jupyter Notebooks**  

   - `agent/CNN-approach/pong.ipynb` & `pong-wrapped.ipynb` — CNN‑based RL experiments  
   - `agent/cv-approach/cv-approach.ipynb` — OpenCV control baseline  
   - `agent/VAE/train_on_dataset.ipynb` — VAE training & visualization  

2. **Latency Logging & Plotting**  
   ```bash
   python utils/plot-latency.py
   ```

3. **Robot Validation Script (Run on the Strech Robot)**  
   ```bash
   python robot/check_lat_com.py
   ```
   See `logs/latency_log_*.csv` for an output example.

## Project Structure

```plaintext
├── agent                   # All agent development code and experiments
│   ├── CNN-approach        # PPO/CNN notebooks for Pong under delay
│   │   ├── pong.ipynb      # Raw environment CNN-based RL notebook
│   │   ├── pong-wrapped.ipynb  # Wrapped environment CNN-based RL notebook
│   │   └── wrapper         # Custom Gym wrappers for delay and observation transforms
│   ├── cv-approach         # OpenCV-based control proof-of-concept
│   │   ├── cv-approach.ipynb  # Notebook applying CV to detect paddle and ball
│   │   └── cv.py           # Script encapsulating CV frame processing logic
│   └── VAE                 # Variational Autoencoder for state representation
│       ├── loss.py         # VAE loss functions
│       ├── train_on_dataset.ipynb  # VAE *offline* training pipeline notebook
│       ├── vae.py          # VAE model implementation
│       └── vae-train.ipynb # VAE *online* training training pipeline notebook
├── logs                    # Latency measurement CSV logs
│   └── latency_log_1748472382.csv  # Example latency log file
├── robot                   # Robot arm communication and validation scripts
│   └── check_lat_com.py    # Tests round-trip communication latency
├── spring-report           # Spring report and associated figures
│   ├── figures             # Generated plots and demo video (see spring-report/repord.md for more context)
│   └── report.md           # Written summary of methodology, results, insights for the spring quarter
└── utils                   # Utility scripts for analysis and plotting
    └── plot-latency.py     # Parses CSV logs and generates latency plots
```

