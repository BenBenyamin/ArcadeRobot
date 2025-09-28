import os
import imageio
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.atari_wrappers import AtariWrapper

import gymnasium as gym

from wrapper import IconOverlayVideoWrapper
from utils import get_icon_config
from utils import ENV_NAME , NO_OP , UP , DOWN

class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        record_freq: int,
        video_path: str,
        fps : int = 60,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.record_freq = record_freq
        self.video_path = video_path
        self.fps = fps
        os.makedirs(self.video_path, exist_ok=True)

        self.env = AtariWrapper(gym.make(ENV_NAME))
        self.video_env = IconOverlayVideoWrapper(
            gym.make(ENV_NAME),
            icon_config=get_icon_config(),
            show_video=False,
            save_video=False
        )
        self.video_env.reset(seed = 0)
        self.env.reset(seed = 0)

    def _on_step(self) -> bool:
        if self.n_calls % self.record_freq != 0: return True
        # 1. Create a unique path for this specific video recording
        video_filename = os.path.abspath(os.path.join(self.video_path, f"{self.num_timesteps}_steps.mp4"))
        self.logger.info(f"Triggering video recording to {video_filename}...")

        # 2. Create the wrapped environment for this session
        # We create it here to pass the unique filename to the wrapper
        

        # 3. Run a full episode with the current model
        reset_out = self.env.reset(seed = 0)
        if isinstance(reset_out, tuple):
            obs, _ = reset_out   # Gymnasium: (obs, info)
        else:
            obs = reset_out      # VecEnv: obs only
        self.video_env.start_recording(video_filename)
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            if action not in [NO_OP,UP,DOWN]:
                action = NO_OP
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.video_env.step(action)
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
                frames = np.array([frame for i, frame in enumerate(video_reader) if i % 16 == 0])
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