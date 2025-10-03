import os
import imageio
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env


import gymnasium as gym

from wrapper import IconOverlayVideoWrapper , PongDelayInertiaWrapper , VectorizedPongDelayInertiaWrapper
from utils import get_icon_config , convert_obs_to_grayscale
from utils import ENV_NAME , NO_OP , UP , DOWN

class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        record_freq: int,
        video_path: str,
        fps : int = 60,
        verbose: int = 1,
        delay:int = 20,
    ):
        super().__init__(verbose)
        self.record_freq = record_freq
        self.video_path = video_path
        self.fps = fps
        os.makedirs(self.video_path, exist_ok=True)

        
        base_env = gym.make(ENV_NAME, render_mode="rgb_array")

        # same preprocessing as training
        base_env = AtariWrapper(base_env, frame_skip=1)

        # Apply frame stacking manually (non-Vec version)
        base_env = gym.wrappers.FrameStackObservation(base_env,stack_size=4)

        if delay > 0:
            base_env= PongDelayInertiaWrapper(base_env,delay_steps=delay)

        self.video_env = IconOverlayVideoWrapper(
            base_env,
            icon_config=get_icon_config(),
            show_video=False,
            save_video=False,
            scale= 4,
            video_fps=5,
        )

        self.video_env.reset(seed = 0)
    def _on_step(self) -> bool:
        if self.n_calls % self.record_freq != 0: return True
        # 1. Create a unique path for this specific video recording
        video_filename = os.path.abspath(os.path.join(self.video_path, f"{self.num_timesteps}_steps.mp4"))
        self.logger.info(f"Triggering video recording to {video_filename}...")

        # 3. Run a full episode with the current model
        obs ,_ = self.video_env.reset(seed = 0)
        self.video_env.start_recording(video_filename)
        done = False
        while not done:
            obs = np.permute_dims(obs,(3,1,2,0))
            action, _ = self.model.predict(obs)
            obs, reward, terminated, truncated, info = self.video_env.step(int(action))
            
            done = terminated or truncated

        # 4. Close the environment. Your wrapper saves the file on close.
        self.video_env.stop_recording()
        self.logger.info(f"Video saved to {video_filename}")

        # 5. Log the saved video file to TensorBoard
        #--- This is the corrected section for getting the writer ---
        try:
            # Find the TensorBoard SummaryWriter in the logger's outputs
            writer = None
            for formatter in self.logger.output_formats:
                if isinstance(formatter, TensorBoardOutputFormat):
                    writer = formatter.writer
                    break
            
            if writer:
                # Read the video file
                video_reader = imageio.get_reader(video_filename)
                frames = np.array([frame for i, frame in enumerate(video_reader)])
                video_reader.close()
                
                # Reshape for PyTorch writer: (N, T, C, H, W)
                frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)
                
                writer.add_video(
                    "trajectory/agent_video",
                    frames_tensor,
                    global_step=self.num_timesteps,
                    fps=self.fps
                )
                self.logger.info("Successfully logged video to TensorBoard.")
            else:
                self.logger.warn("Could not find TensorBoard writer.")

        except Exception as e:
            self.logger.error(f"Failed to log video: {e}")

        return True