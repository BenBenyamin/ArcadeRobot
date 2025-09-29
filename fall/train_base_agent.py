import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3 import PPO , DQN
from callback import VideoRecorderCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


import ale_py
gym.register_envs(ale_py)

ENV_NAME = "PongNoFrameskip-v4"
TOTAL_TIMESTEPS = 1_500_000
MODEL_PATH = f"ppo_pong_vanilla"
BATCH_SIZE = 256
DELAY = 0

env = make_atari_env(ENV_NAME,n_envs=32)
env = VecFrameStack(env, n_stack=4)


# env = VectorizedPongDelayInertiaWrapper(env,DELAY)

val_env = gym.make(ENV_NAME, render_mode="rgb_array")
val_env = AtariWrapper(val_env)
val_env = Monitor(val_env) 

video_callback = VideoRecorderCallback(record_freq=8*256,video_path="./pong_videos",verbose=0,delay=DELAY)
 
paitence_callback  = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=60, verbose=0)
eval_callback = EvalCallback(val_env, eval_freq=10240, callback_after_eval=paitence_callback, verbose=0)

callback = CallbackList([video_callback,eval_callback])

model = PPO("CnnPolicy", env, verbose=1 , device="cuda",tensorboard_log="./tensorboard/", batch_size=BATCH_SIZE)
# model = PPO.load(MODEL_PATH, env)

model.learn(TOTAL_TIMESTEPS, callback= video_callback)

model.save(MODEL_PATH)

env.close()