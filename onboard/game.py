import numpy as np
import gymnasium as gym
import pygame
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
import ale_py
gym.register_envs(ale_py)

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


ENV_NAME = "PongNoFrameskip-v4"

class GameEnv:
    def __init__(self,fps: int = 30, scale: int = 2, seed = None):
        pygame.init()

        # --- Environment setup ---
        self.env = make_vec_env(
                ENV_NAME, n_envs=1, 
                wrapper_class=AtariWrapper,
                wrapper_kwargs={"frame_skip": 1},
                seed = seed,
            )
        self.out_frames = self.env.reset()

        # --- Get first frame for dimensions ---
        frame = self.env.render()
        self.h, self.w, _ = frame.shape
        self.scale = scale
        self.fps = fps

        # --- Pygame display ---
        self.screen = pygame.display.set_mode((self.w * scale, self.h * scale))
        pygame.display.set_caption("Pong (Pygame Render)")
        self.clock = pygame.time.Clock()

        # --- Icon setup ---
        icon_size = (30 * scale, 24 * scale)
        self.icons_map = {
            0: pygame.transform.scale(pygame.image.load("./ICONS/NO_OP.png").convert_alpha(), icon_size),
            2: pygame.transform.scale(pygame.image.load("./ICONS/UP.png").convert_alpha(), icon_size),
            3: pygame.transform.scale(pygame.image.load("./ICONS/DOWN.png").convert_alpha(), icon_size),
        }

        self.action = 0
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(1)
            self.joystick.init()

    def _embedd_icon(self, action, surface):
        """Draws current action icon on surface."""
        icon = self.icons_map.get(int(action))
        if icon:
            pos = (130 * self.scale, 0)
            surface.blit(icon, pos)

    def render(self):
        """Render one frame at target FPS."""
        frame = self.env.render()
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (self.w * self.scale, self.h * self.scale))
        self._embedd_icon(self.action, surf)
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.fps)  # maintain FPS
        return frame

    def _read_action(self) -> np.ndarray:
        """Read UP/DOWN/NOOP from keyboard or joystick."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        keys = pygame.key.get_pressed()
        action = 0  # NOOP by default

        if keys[pygame.K_UP]:
            action = 2
        elif keys[pygame.K_DOWN]:
            action = 3

        if self.joystick:
            hat_x, hat_y = self.joystick.get_hat(0)
            if hat_y == 1:
                action = 2
            elif hat_y == -1:
                action = 3

        self.action = action
        return np.array([action])

    def step(self):
        """Perform one environment step and render."""
        action = self._read_action()
        self.out_frames, rewards, dones, infos = self.env.step(action)
        self.render()
        if np.any(dones):
            self.env.reset()
        return self.out_frames

