import gymnasium as gym
from collections import deque
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from utils import alpha_blit_rgb , load_icon_and_resize
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
        scale: int = 8,
        waitkey_ms: int = 1,
        save_video: bool = True,
        video_path: str = "video.mp4",
        video_fps: int = 30,
        video_codec: str = "mp4v",  # try "avc1" if available
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
        icon = self._icons.get(action)
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
        obs, info = self.env.reset(**kwargs)  # Gymnasium API
        if self.overlay_on_reset:
            obs = self._overlay_icon_for_action(obs, self.reset_action_for_overlay)
        self._display_and_maybe_write(obs)
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._overlay_icon_for_action(obs, action)
        self._display_and_maybe_write(obs)
        return obs, reward, terminated, truncated, info

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
    action. This wrapper is UI-agnostic; it does not draw or record anything.
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
        if action not in (NO_OP, UP, DOWN):
            action = NO_OP

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




