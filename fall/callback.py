import os
import imageio
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env
import torch.nn.functional as F

import gymnasium as gym

from wrapper import IconOverlayVideoWrapper , PongDelayInertiaWrapper , VectorizedActionLoggerWrapper
from utils import get_icon_config
from utils import ENV_NAME , NO_OP , UP , DOWN

class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        record_freq: int,
        video_path: str,
        fps : int = 60,
        verbose: int = 1,
        delay:int = 20,
        stack_size:int = 4,
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
        if stack_size > 0:
            base_env = gym.wrappers.FrameStackObservation(base_env,stack_size=stack_size)
        
        self.stack_size = stack_size

        self.video_env = IconOverlayVideoWrapper(
            base_env,
            icon_config=get_icon_config(),
            show_video=False,
            save_video=False,
            scale= 4,
            video_fps=60,
        )

        if delay > 0:
            self.video_env= PongDelayInertiaWrapper(self.video_env,delay_steps=delay)

        self.video_env.reset(seed = 0)
    def _on_step(self) -> bool:
        if self.n_calls % self.record_freq != 0: return True
        # 1. Create a unique path for this specific video recording
        video_filename = os.path.abspath(os.path.join(self.video_path, f"{self.num_timesteps}_steps.mp4"))
        self.logger.info(f"Triggering video recording to {video_filename}...")

        # 3. Run a full episode with the current model
        obs ,_ = self.video_env.reset(seed = 0)
        self.video_env.env.start_recording(video_filename)
        done = False
        while not done: 
            if self.stack_size > 0:
                obs = np.permute_dims(obs,(3,1,2,0))
            action, _ = self.model.predict(obs)
            obs, reward, terminated, truncated, info = self.video_env.step(int(action))
            
            done = terminated or truncated

        # 4. Close the environment. Your wrapper saves the file on close.
        self.video_env.env.stop_recording()
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
                frames = []
                i = 0
                for f in video_reader:
                    if i % 8 == 0:
                        frames.append(f)
                    i+=1
                frames = np.array(frames,dtype=np.uint8)
                video_reader.close()
                
                # Reshape for PyTorch writer: (N, T, C, H, W)
                frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)
                B, T, C, H, W = frames_tensor.shape

                # Flatten time into batch dimension: (B*T, C, H, W)
                frames_2d = frames_tensor.view(-1, C, H, W).float()

                # Compute target resolution
                target_height = 240
                target_width = int(W * target_height / H)

                # Downsample all frames
                frames_scaled = F.interpolate(
                    frames_2d,
                    size=(target_height, target_width),
                    mode="area"
                ).to(torch.uint8)

                # Reshape back to (1, T, C, H', W')
                frames_tensor = frames_scaled.view(B, T, C, target_height, target_width)

                print(f"[VideoRecorderCallback] Logging video with shape {frames_tensor.shape}, dtype={frames_tensor.dtype}")
                
                writer.add_video(
                    "trajectory/agent_video",
                    frames_tensor,
                    global_step=self.model.num_timesteps,
                    fps=self.fps
                )
                writer.flush()
                self.logger.info("Successfully logged video to TensorBoard.")
            else:
                self.logger.warn("Could not find TensorBoard writer.")

        except Exception as e:
            self.logger.error(f"Failed to log video: {e}")

        return True


class OneTimeAttachLoggerCallback(BaseCallback):
    """
    Runs exactly once at the start of training.
    Attaches the SB3 TensorBoard logger to the action logger wrapper.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_training_start(self):
        # Runs once when .learn() begins
        VectorizedActionLoggerWrapper.attach_logger(self.training_env, self.logger)
        if self.verbose > 0:
            print("[OneTimeAttachLoggerCallback] Logger attached to ActionLoggerWrapper.")

    def _on_step(self):
        # Do nothing during steps
        return True