import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.atari_wrappers import AtariWrapper
from collections import deque
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from stable_baselines3.common.logger import TensorBoardOutputFormat
import matplotlib.pyplot as plt
import io
from PIL import Image
from utils import ALE_ACTION_MAP

from utils import alpha_blit_rgb , load_icon_and_resize , convert_obs_to_grayscale
from utils import UP, DOWN, NO_OP


class IconOverlayVideoWrapper(gym.Wrapper):
    """
    Generic wrapper that:
      - overlays per-action icons onto RGB observations,
      - optionally shows a live OpenCV window (resized),
      - optionally records frames to a video file.

    IMPORTANT: Place this wrapper *below* any higher-level wrappers that internally
    call env.step (e.g., your inertia wrapper). That way, it can see/record those
    internal steps too.

    Example composition:
        env = gym.make(ENV_NAME, render_mode=None)  # or "rgb_array"
        env = IconOverlayVideoWrapper(env, icon_config=..., ...)
        env = PongDelayInertiaWrapper(env, delay_steps=10)
    """

    def __init__(
        self,
        env,
        icon_config: Dict[int, Tuple[str, int]],
        icon_xy: Tuple[int, int] = (130, 0),
        show_video: bool = True,
        window_name: str = "Pong",
        scale: int = 1,
        waitkey_ms: int = 1,
        save_video: bool = True,
        video_path: str = "video.mp4",
        video_fps: int = 30,
        video_codec: str = "mp4v",
        overlay_on_reset: bool = True,
        reset_action_for_overlay: int = NO_OP,
    ):
        super().__init__(env)
        self.icon_xy = icon_xy
        self.show_video = show_video
        self.window_name = window_name
        self.scale = int(scale)
        self.waitkey_ms = int(waitkey_ms)

        self.save_video = save_video
        self.video_path = video_path
        self.video_fps = int(video_fps)
        self.video_codec = video_codec
        self._writer: Optional[cv2.VideoWriter] = None

        self.overlay_on_reset = overlay_on_reset
        self.reset_action_for_overlay = reset_action_for_overlay

        # Load icons now
        self._icons: Dict[int, Optional[np.ndarray]] = {}
        for action, (path, height) in icon_config.items():
            self._icons[action] = load_icon_and_resize(path, target_height=height)

    # ---- Core helpers ----
    def _overlay_icon_for_action(self, obs: np.ndarray, action: int) -> np.ndarray:
        icon = self._icons.get(int(action))
        alpha_blit_rgb(obs, icon, 255, self.icon_xy[0], self.icon_xy[1])
        return obs

    def _ensure_writer(self, frame_bgr: np.ndarray):
        if not self.save_video or self._writer is not None:
            return
        h, w = frame_bgr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        self._writer = cv2.VideoWriter(self.video_path, fourcc, self.video_fps, (w, h))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter at: {self.video_path}")

    def _display_and_maybe_write(self, obs_rgb: np.ndarray):
        # Convert to BGR for OpenCV and resize for display/recording
        frame = cv2.cvtColor(obs_rgb, cv2.COLOR_RGB2BGR)
        if self.scale != 1:
            frame = cv2.resize(
                frame,
                (frame.shape[1] * self.scale, frame.shape[0] * self.scale),
                interpolation=cv2.INTER_NEAREST,
            )

        if self.show_video:
            cv2.imshow(self.window_name, frame)
            # Keep window responsive / allow 'q' to quit viewer (doesn't close env)
            if cv2.waitKey(self.waitkey_ms) & 0xFF == ord("q"):
                self.show_video = False
                cv2.destroyWindow(self.window_name)

        if self.save_video:
            self._ensure_writer(frame)
            self._writer.write(frame)

    # ---- Gym API ----
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)  # agent obs (grayscale if wrapped)

        # always grab RGB from emulator for video
        try:
            rgb = self.env.unwrapped.render()                # Gymnasium (render_mode="rgb_array")
        except TypeError:
            rgb = self.env.unwrapped.render(mode="rgb_array")  # Classic Gym fallback

        if self.overlay_on_reset:
            obs_rgb = self._overlay_icon_for_action(rgb, self.reset_action_for_overlay)
        else:
            obs_rgb = rgb

        self._display_and_maybe_write(obs_rgb)
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # always grab RGB for video
        try:
            rgb = self.env.unwrapped.render()
        except TypeError:
            rgb = self.env.unwrapped.render(mode="rgb_array")

        obs_rgb = self._overlay_icon_for_action(rgb, action)
        self._display_and_maybe_write(obs_rgb)

        return obs, reward, terminated, truncated, info



    def start_recording(self, video_path: str) -> None:
        """Starts the video recording process."""
        self.video_path = video_path
        self.save_video = True
        # The _ensure_writer method will be called on the next step

    def stop_recording(self) -> None:
        """Calls the save method to finalize the video."""
        self.save()
        self.save_video = False
    
    def save(self) -> None:
        """
        Finishes and saves the current video file.
        """
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def close(self):
        try:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
        finally:
            # Only destroy window if we created one and it's still open
            if self.show_video:
                try:
                    cv2.destroyWindow(self.window_name)
                except Exception:
                    pass
            cv2.destroyAllWindows()
        return super().close()



class PongDelayWrapper(gym.Wrapper):

    def __init__(self, env, delay_steps=1):
        super().__init__(env)

        self.buffer_stack_size = delay_steps

        self.action_queue = deque(
            [NO_OP] * delay_steps, 
            maxlen=delay_steps
        )

    def reset(self, **kwargs):

        obs = self.env.reset(**kwargs)
        self.action_queue.clear()
        self.action_queue.extend([NO_OP] * self.buffer_stack_size)

        return obs


    def step(self,action):

        delayed_action = self.action_queue.popleft()
        obs, reward, terminated, truncated, info = self.env.step(delayed_action)
        self.action_queue.append(action)

        return obs, reward, terminated, truncated, info


class PongDelayInertiaWrapper(gym.Wrapper):
    """
    Adds inertia behavior to discrete actions (NO_OP, UP, DOWN) by inserting
    internal steps with previous/NO_OP actions before executing the requested
    action.
    """

    def __init__(self, env, delay_steps: int = 10):
        super().__init__(env)
        self.delay_steps = int(delay_steps)
        self.prev_action = NO_OP

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)  # Gymnasium API
        self.prev_action = NO_OP
        return obs, info

    def _run_steps(self, action: int, n_steps: int, total_reward: float):
        """
        Advance n_steps internally, accumulating rewards, and break if episode ends.
        Returns: obs, total_reward, terminated, truncated, info, done_flag
        """
        obs, reward, terminated, truncated, info = None, 0.0, False, False, {}
        for _ in range(n_steps):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                return obs, total_reward, terminated, truncated, info, True
        return obs, total_reward, terminated, truncated, info, False

    def step(self, action: int):
        total_reward = 0.0
        # if action not in (NO_OP, UP, DOWN):
        #     action = NO_OP

        if self.prev_action != action:
            if action == NO_OP:
                # Coasting inertia: half delay on previous action
                obs, total_reward, terminated, truncated, info, done = \
                    self._run_steps(self.prev_action, self.delay_steps // 2, total_reward)
                if done:
                    return obs, total_reward, terminated, truncated, info
            else:
                # First half: previous action
                obs, total_reward, terminated, truncated, info, done = \
                    self._run_steps(self.prev_action, self.delay_steps // 2, total_reward)
                if done:
                    return obs, total_reward, terminated, truncated, info
                # Second half: NO_OP
                obs, total_reward, terminated, truncated, info, done = \
                    self._run_steps(NO_OP, self.delay_steps // 2, total_reward)
                if done:
                    return obs, total_reward, terminated, truncated, info

        # Finally, execute desired action
        self.prev_action = action
        obs, reward, terminated, truncated, info = self.env.step(action)
        total = float(reward) + total_reward
        return obs, total, terminated, truncated, info


class VectorizedPongDelayInertiaWrapper(VecEnvWrapper):

    def __init__(self, venv, delay_steps: int = 10):
        super().__init__(venv)
        self.delay_steps = int(delay_steps)
        self.prev_actions = np.full(self.num_envs, NO_OP, dtype=np.int64)
        self._desired_actions = None

    def reset(self,**kwargs):

        obs = self.venv.reset(**kwargs)
        self.prev_actions = np.full(self.num_envs, NO_OP, dtype=np.int64)
        self._desired_actions = None

        return obs
    
    def step_async(self, actions):
        # return super().step_async(actions)
        actions = np.asarray(actions, dtype=np.int64)
        # actions[~np.isin(actions, [NO_OP, UP, DOWN])] = NO_OP
        self._desired_actions = actions
        

    def _run_steps(self, env_id: int, action: int, n_steps: int, total_reward: float):
        env = self.venv.envs[env_id]
        obs, reward, terminated, truncated, info = None, 0.0, False, False, {}

        for _ in range(n_steps):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                obs, reset_info = env.reset()
                info["terminal_observation"] = obs
                info["reset_info"] = reset_info
                self.prev_actions[env_id] = NO_OP
                return obs, total_reward, terminated, truncated, info, True

        return obs, total_reward, terminated, truncated, info, False

    def step_wait(self):
        batch_obs, batch_rewards, batch_dones, batch_infos = [], [], [], []

        for i in range(self.num_envs):
            desired = self._desired_actions[i]
            prev = self.prev_actions[i]
            total_reward = 0.0
            obs, info, terminated, truncated, done = None, {}, False, False, False

            if desired != prev and desired in (NO_OP,UP,DOWN):
                if desired == NO_OP:
                    obs, total_reward, terminated, truncated, info, done = \
                        self._run_steps(i, prev, self.delay_steps // 2, total_reward)
                else:
                    obs, total_reward, terminated, truncated, info, done = \
                        self._run_steps(i, prev, self.delay_steps // 2, total_reward)
                    if not done:
                        obs, total_reward, terminated, truncated, info, done = \
                            self._run_steps(i, NO_OP, self.delay_steps // 2, total_reward)

            if not done:
                obs, reward, terminated, truncated, info = self.venv.envs[i].step(desired)
                total_reward += float(reward)

            if terminated or truncated:
                final_obs, reset_info = self.venv.envs[i].reset()
                info["terminal_observation"] = obs
                info["reset_info"] = reset_info
                obs = final_obs
                self.prev_actions[i] = NO_OP
            else:
                self.prev_actions[i] = desired

            # Collect results
            batch_obs.append(obs)
            batch_rewards.append(total_reward)
            batch_dones.append(terminated or truncated)
            batch_infos.append(info)

        return (
            np.stack(batch_obs),
            np.array(batch_rewards, dtype=np.float32),
            np.array(batch_dones, dtype=bool),
            batch_infos,
        )

class VectorizedActionLoggerWrapper(VecEnvWrapper):
    """
    Logs agent actions to TensorBoard when an episode ends (done or truncated),
    and includes a Matplotlib bar-plot image of action frequencies with
    ALE action names as x-axis labels.
    """

    def __init__(self, venv, logger=None, tag_prefix: str = "policy",):
        super().__init__(venv)
        self.logger = logger
        self.tag_prefix = tag_prefix

        self._actions_per_env = [[] for _ in range(self.num_envs)]
        self._episodes_logged = 0
        self._tb_writer = None

    # ----------------- VecEnv interface -----------------

    def step_async(self, actions):
        """Record chosen actions from the policy for each sub-env."""
        if isinstance(actions, np.ndarray):
            acts = actions.reshape(-1)
        elif isinstance(actions, (list, tuple)):
            acts = np.array(actions).reshape(-1)
        else:
            acts = np.array([actions])

        for i in range(self.num_envs):
            self._actions_per_env[i].append(int(acts[i] if acts.size > 1 else acts[0]))

        self.venv.step_async(actions)   

    def step_wait(self):
        """When an episode ends, log its action histogram."""
        obs, rewards, dones, infos = self.venv.step_wait()

        for i, done in enumerate(dones):
            if done:
                self._log_episode_actions(i)
                self._actions_per_env[i].clear()

        return obs, rewards, dones, infos

    def reset(self):
        return self.venv.reset()

    # ----------------- Logging helpers -----------------

    def _ensure_tb_writer(self):
        """Fetch and cache the TensorBoard writer from the SB3 logger."""
        if self._tb_writer is not None:
            return self._tb_writer
        if self.logger is None:
            return None

        for fmt in self.logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                self._tb_writer = fmt.writer
                break
        return self._tb_writer

    def _log_episode_actions(self, env_idx: int):
        """Log Matplotlib bar chart of action frequencies with ALE names."""
        writer = self._ensure_tb_writer()
        buf = self._actions_per_env[env_idx]
        if writer is None or not buf:
            return

        actions_np = np.asarray(buf, dtype=np.int32)
        self._episodes_logged += 1
        step = self._episodes_logged // 300

        # --- Compute action frequencies ---
        unique, counts = np.unique(actions_np, return_counts=True)

        # Map actions to names, fall back to numeric if unknown
        labels = [ALE_ACTION_MAP.get(int(a), str(a)) for a in unique]

        # --- Create Matplotlib bar chart ---
        plt.figure(figsize=(6, 6))
        plt.bar(labels, counts, color="#1f77b4", edgecolor="black")
        plt.title(f"Action Frequency (Episode) Step = {step}")
        plt.xlabel("Action")
        plt.ylabel("Count")
        plt.xticks(ha="center")
        plt.tight_layout()

        # --- Convert figure to image array (HWC) ---
        buf_img = io.BytesIO()
        plt.savefig(buf_img, format="png", bbox_inches="tight")
        plt.close()
        buf_img.seek(0)
        img = np.array(Image.open(buf_img))
        buf_img.close()

        # --- Log to TensorBoard Images tab ---
        writer.add_image(
            f"{self.tag_prefix}/actions_bar_plot",
            img,
            global_step=step,
            dataformats="HWC",
        )
        writer.flush()

    # ----------------- Public utilities -----------------

    def set_logger(self, logger):
        """Attach SB3-style logger at runtime."""
        self.logger = logger
        self._tb_writer = None  # reset cache so we pick up new writer

    @staticmethod
    def attach_logger(env, logger) -> bool:
        """Traverse VecEnvWrapper chain and attach logger."""
        current = env
        while isinstance(current, VecEnvWrapper):
            if isinstance(current, VectorizedActionLoggerWrapper):
                current.set_logger(logger)
                return True
            current = current.venv
        return False
