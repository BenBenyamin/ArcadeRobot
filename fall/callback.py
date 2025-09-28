import os
import imageio
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        record_freq: int,
        video_path: str,
        fps : int = 60,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.record_freq = record_freq
        self.video_path = video_path
        self.fps = fps
        os.makedirs(self.video_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.record_freq != 0: return True
        # 1. Create a unique path for this specific video recording
        video_filename = os.path.abspath(os.path.join(self.video_path, f"{self.num_timesteps}_steps.mp4"))
        self.logger.info(f"Triggering video recording to {video_filename}...")

        # 2. Create the wrapped environment for this session
        # We create it here to pass the unique filename to the wrapper
        

        # 3. Run a full episode with the current model
        obs = self.training_env.reset()
        self.training_env.env_method("start_recording", video_filename)
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, done, infos = self.training_env.step(action)

        # 4. Close the environment. Your wrapper saves the file on close.
        self.training_env.env_method("stop_recording")
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
                    fps=60
                )
                self.logger.info("Successfully logged video to TensorBoard.")
            else:
                self.logger.warn("Could not find TensorBoard writer.")

        except Exception as e:
            self.logger.error(f"Failed to log video: {e}")

        return True