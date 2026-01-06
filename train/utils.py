
import os
import cv2
import gymnasium as gym
import numpy as np
import ale_py
from typing import  Optional
import sys


# ---- Register ALE envs (needed in some setups) ----
gym.register_envs(ale_py)



# --- ALE Action Map for Pong and other Atari games ---
# Reference: https://ale.farama.org/environments/pong/
ALE_ACTION_MAP = {
    0: "NO-OP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT",
    4: "RIGHTFIRE",
    5: "LEFTFIRE",
}

_REMAPPING = {
    "NO-OP"     : "NO-OP",
    "FIRE"      : "NO-OP",
    "RIGHT"     : "RIGHT",
    "LEFT"      : "LEFT",
    "RIGHTFIRE" : "RIGHT",
    "LEFTFIRE"  : "LEFT",
}

# GET THE NUMBERS
ALE_EFFECTIVE_ACTION_MAP = {
    k: next(
        num for num, name in ALE_ACTION_MAP.items() 
        if name == _REMAPPING[v]
    )
    for k, v in ALE_ACTION_MAP.items()
}

REV_ACTION_MAP = {v: k for k, v in ALE_ACTION_MAP.items()}

# ---- Actions ----
NO_OP = REV_ACTION_MAP["NO-OP"]
UP = REV_ACTION_MAP["RIGHT"]
DOWN = REV_ACTION_MAP["LEFT"]

ENV_NAME = "PongNoFrameskip-v4"


def convert_obs_to_grayscale(obs: np.ndarray, h: int = 84, w: int = 84) -> np.ndarray:

    frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return frame[..., None] 

def get_icon_config():

    # Resolve icon paths (adjust this to your layout)
    if "ipykernel" in sys.modules:
        script_dir = os.getcwd()
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    icon_dir = os.path.join(script_dir, "icon")
    icon_config = {
        NO_OP: (os.path.join(icon_dir, "NO_OP.png"), 18),
        UP:    (os.path.join(icon_dir, "UP.png"),    24),
        DOWN:  (os.path.join(icon_dir, "DOWN.png"),  24),
    }

    for action in ALE_ACTION_MAP.keys():
        if action in (NO_OP,UP,DOWN): continue

        icon_config[action] = (os.path.join(icon_dir, f"{action}.png"),    24)

    return icon_config

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
