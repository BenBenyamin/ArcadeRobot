from game import GameEnv
from control import StretchControl
from sb3_contrib import QRDQN
from stable_baselines3 import DQN , PPO
from time import sleep

action_remapping = {
    0: 0.03,  # NO OP
    1: 0.04,  # UP
    2: 0.02   # DOWN
}
## 55
game_env = GameEnv(fps=30, scale=4 ,seed= 5)
controller = StretchControl(
                            algorithm= PPO,
                            model_path="PPO_stochastic_268160000_steps",
                            action_remapping=action_remapping,
                            )

sleep(3)
while True:

    game_env.step()

    controller.command(obs=game_env.out_frames)


