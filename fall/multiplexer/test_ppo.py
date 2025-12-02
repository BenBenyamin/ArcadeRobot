import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env , make_atari_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import torch
from ppo import PPO

def main():
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


    # -------------------------
    #   CONFIG
    # -------------------------
    ENV_ID = "Hopper-v5"
    NUM_ENVS = 1
    NUM_TIMESTEPS = 500_000

    # -------------------------
    #   ENV SETUP
    # -------------------------
    env = make_vec_env(ENV_ID, n_envs=NUM_ENVS)
    env = gym.make(ENV_ID)

    # -------------------------
    #   MODEL SETUP
    # -------------------------
    ppo = PPO(
        env=env,
        device= "cuda:0",
        hidden_size=64,
        num_steps=1024,
        num_minibatches=4,
        update_epochs=4,
        learning_rate=3e-4,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        kl_coeff=0.0,
        target_kl=0.01,
        anneal_lr=False,
    )

    print("\n===== PPO TEST STARTED =====\n")

    # -------------------------
    #   EVALUATION FUNCTION
    # -------------------------
    def evaluate(env, agent, num_episodes=5):
        scores = []
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated

            scores.append(total_reward)

        return np.mean(scores)

    eval_env = gym.make(ENV_ID)

    # -------------------------
    #   TRAINING LOOP
    # -------------------------
    total_target = 0
    for iteration in range(1, NUM_TIMESTEPS // 1024):
        total_target += 1024 * NUM_ENVS
        ppo.learn(total_timesteps=total_target)

        if iteration % 10 == 0:
            score = evaluate(eval_env, ppo)
            print(f"[TIMESTEP {total_target}] Mean Reward = {score:.5f}")

    print("\n===== TEST FINISHED =====\n")


if __name__ == "__main__":
    main()
