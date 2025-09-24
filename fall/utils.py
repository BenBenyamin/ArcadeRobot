
import os
import cv2
import gymnasium as gym
import numpy as np
import ale_py
from typing import Dict, Optional, Tuple

# ---- Register ALE envs (needed in some setups) ----
gym.register_envs(ale_py)

# ---- Actions ----
NO_OP = 0
UP = 2
DOWN = 5

ENV_NAME = "PongNoFrameskip-v4"

def make_pong_env(
    video_wrapper:gym.Wrapper, 
    delay_wrapper:gym.Wrapper,
    delay_steps: int = 20,
    train: bool = True,  # when True, don't spawn ALE SDL window; we show via OpenCV
    show_video: bool = True,
    save_video: bool = True,
    video_path: str = "video.mp4",
    video_fps: int = 30,
    icon_xy: Tuple[int, int] = (130, 0),
    scale: int = 8,
    waitkey_ms: int = 1,
):
    """
    Build an environment with:
      base ALE → IconOverlayVideoWrapper → PongDelayInertiaWrapper
    This order ensures the overlay/recorder sees all internal steps.
    """
    # Resolve icon paths (adjust this to your layout)
    icon_dir = os.path.join(os.getcwd(), "icon")
    icon_config = {
        NO_OP: (os.path.join(icon_dir, "NO_OP.png"), 18),
        UP:    (os.path.join(icon_dir, "UP.png"),    24),
        DOWN:  (os.path.join(icon_dir, "DOWN.png"),  24),
    }

    def _init():
        base = gym.make(ENV_NAME, render_mode=None if train else "human")
        base = video_wrapper(
            base,
            icon_config=icon_config,
            icon_xy=icon_xy,
            show_video=show_video,
            window_name="Pong",
            scale=scale,
            waitkey_ms=waitkey_ms,
            save_video=save_video,
            video_path=video_path,
            video_fps=video_fps,
            video_codec="mp4v",
            overlay_on_reset=True,
            reset_action_for_overlay=NO_OP,
        )
        env = delay_wrapper(base, delay_steps=delay_steps)
        return env

    return _init

def load_icon_and_resize(path: str, target_height: int = 24) -> Optional[np.ndarray]:
    """
    Load an icon from disk (preserving alpha), convert to RGB(A), and resize by height.
    Returns np.uint8 image (H, W, 3 or 4) or None if not found.
    """
    icon = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if icon is None:
        print(f"[icon load warning] Could not read: {path}")
        return None

    # Convert BGR/ BGRA -> RGB/ RGBA
    if icon.ndim == 2:
        icon = cv2.cvtColor(icon, cv2.COLOR_GRAY2RGB)
    elif icon.shape[2] == 3:
        icon = cv2.cvtColor(icon, cv2.COLOR_BGR2RGB)
    elif icon.shape[2] == 4:
        icon = cv2.cvtColor(icon, cv2.COLOR_BGRA2RGBA)

    # Resize by height
    h, w = icon.shape[:2]
    if h != target_height:
        scale = target_height / float(h)
        new_w = max(1, int(round(w * scale)))
        icon = cv2.resize(icon, (new_w, target_height), interpolation=cv2.INTER_AREA)

    return icon


def alpha_blit_rgb(dst_rgb: np.ndarray, src_img: Optional[np.ndarray], alpha: int, x: int, y: int) -> np.ndarray:
    """
    Paste src_img (RGB or RGBA) onto dst_rgb (RGB) at (x, y).
    Uses per-pixel alpha if src has one; otherwise uses uniform alpha (0..255).
    Modifies dst in place and also returns it.
    """
    if dst_rgb is None or src_img is None:
        return dst_rgb

    H, W = dst_rgb.shape[:2]
    h, w = src_img.shape[:2]

    if x >= W or y >= H:
        return dst_rgb
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    w_clip = x2 - x
    h_clip = y2 - y
    if w_clip <= 0 or h_clip <= 0:
        return dst_rgb

    dst_roi = dst_rgb[y:y2, x:x2].astype(np.float32)

    if src_img.shape[2] == 4:
        # Per-pixel alpha
        src_rgb = src_img[:h_clip, :w_clip, :3].astype(np.float32)
        src_a = src_img[:h_clip, :w_clip, 3:4].astype(np.float32) / 255.0
        out = src_a * src_rgb + (1.0 - src_a) * dst_roi
    else:
        # Uniform alpha
        src_rgb = src_img[:h_clip, :w_clip, :3].astype(np.float32)
        a = float(alpha) / 255.0
        out = a * src_rgb + (1.0 - a) * dst_roi

    dst_rgb[y:y2, x:x2] = np.clip(out, 0, 255).astype(np.uint8)
    return dst_rgb
