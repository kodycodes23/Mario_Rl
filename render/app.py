"""Entry point: menu / human-play / watch-agent modes. The only place both env and render meet.

`env/`, `game/`, and `agent/train.py` must never import this module (see the performance
boundary in mario_rl_build_prompt.md §7) -- only `agent/play.py`'s `--render` path and this
file's own `__main__` import pygame at all.
"""
from __future__ import annotations

import argparse
import enum
import sys
from pathlib import Path

import numpy as np
import pygame

from game.constants import DT, FPS, MAX_EPISODE_STEPS, Action
from game.tilemap import TileMap
from game.world import World
from render import theme, ui
from render.renderer import PX_PER_TILE, Renderer

SCREEN_W, SCREEN_H = 960, 540
FADE_SPEED = 3.0  # fade alpha units / second


class AppState(enum.Enum):
    MENU = "menu"
    LEVEL_SELECT = "level_select"
    HUMAN_PLAY = "human_play"
    WATCH_AGENT = "watch_agent"
    SETTINGS = "settings"


LEVELS = {
    "Level 1 (training)": "game/levels/level_1.json",
    "Level 2 (held-out)": "game/levels/level_2.json",
    "Level 3 (medium)": "game/levels/level_3.json",
    "Level 4 (hard)": "game/levels/level_4.json",
    "Level 5 (hardest)": "game/levels/level_5.json",
}


def _make_fonts():
    return (
        pygame.font.Font(theme.FONT_NAME, theme.FONT_SIZE_SMALL),
        pygame.font.Font(theme.FONT_NAME, theme.FONT_SIZE_MED),
        pygame.font.Font(theme.FONT_NAME, theme.FONT_SIZE_LARGE),
    )


def find_latest_checkpoint(runs_dir: str = "runs") -> str | None:
    """Newest best_model.zip (falling back to final_model.zip) under runs/*/, by mtime."""
    root = Path(runs_dir)
    if not root.is_dir():
        return None
    candidates = list(root.glob("*/best_model.zip")) + list(root.glob("*/final_model.zip"))
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def _keys_to_action(keys) -> Action:
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    jump = keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w]
    if left and jump:
        return Action.LEFT_JUMP
    if right and jump:
        return Action.RIGHT_JUMP
    if left:
        return Action.LEFT
    if right:
        return Action.RIGHT
    if jump:
        return Action.JUMP
    return Action.NOOP


class App:
    """Menu-driven app: start, level select, human play, watch agent, settings."""

    def __init__(self, screen: pygame.Surface, clock: pygame.time.Clock,
                 default_checkpoint: str | None = None):
        self.screen = screen
        self.clock = clock
        self.fonts = _make_fonts()
        self.state = AppState.MENU
        self.menu_options = ["Human play", "Watch agent", "Level select", "Quit"]
        self.menu_selected = 0
        self.level_options = list(LEVELS.keys())
        self.level_selected = 0
        self.selected_level_path = LEVELS[self.level_options[0]]
        self.default_checkpoint = default_checkpoint or find_latest_checkpoint()
        self.fade_alpha = 0.0
        self.fade_target_state: AppState | None = None
        self.running = True

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self._handle_events()
            self._update(dt)
            self._draw()
            pygame.display.flip()

    def _goto(self, state: AppState):
        self.fade_target_state = state
        self.fade_alpha = 1e-6  # kick off fade-out

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if self.state == AppState.MENU:
                    self._handle_menu_key(event.key)
                elif self.state == AppState.LEVEL_SELECT:
                    self._handle_level_select_key(event.key)
                elif event.key == pygame.K_ESCAPE:
                    self._goto(AppState.MENU)

    def _handle_menu_key(self, key):
        if key in (pygame.K_UP, pygame.K_w):
            self.menu_selected = (self.menu_selected - 1) % len(self.menu_options)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.menu_selected = (self.menu_selected + 1) % len(self.menu_options)
        elif key == pygame.K_RETURN:
            choice = self.menu_options[self.menu_selected]
            if choice == "Human play":
                self._goto(AppState.HUMAN_PLAY)
            elif choice == "Watch agent":
                self._goto(AppState.WATCH_AGENT)
            elif choice == "Level select":
                self._goto(AppState.LEVEL_SELECT)
            elif choice == "Quit":
                self.running = False
        elif key == pygame.K_ESCAPE:
            self.running = False

    def _handle_level_select_key(self, key):
        if key in (pygame.K_UP, pygame.K_w):
            self.level_selected = (self.level_selected - 1) % len(self.level_options)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.level_selected = (self.level_selected + 1) % len(self.level_options)
        elif key == pygame.K_RETURN:
            self.selected_level_path = LEVELS[self.level_options[self.level_selected]]
            self._goto(AppState.MENU)
        elif key == pygame.K_ESCAPE:
            self._goto(AppState.MENU)

    def _update(self, dt: float):
        if self.fade_target_state is not None:
            self.fade_alpha += FADE_SPEED * dt
            if self.fade_alpha >= 1.0:
                self.state = self.fade_target_state
                self.fade_target_state = None
                if self.state == AppState.HUMAN_PLAY:
                    self._run_human_play_session()
                elif self.state == AppState.WATCH_AGENT:
                    self._run_watch_agent_session()
        else:
            self.fade_alpha = max(0.0, self.fade_alpha - FADE_SPEED * dt)

    def _draw(self):
        if self.state == AppState.MENU:
            ui.draw_menu(self.screen, self.fonts, self.menu_options, self.menu_selected)
        elif self.state == AppState.LEVEL_SELECT:
            ui.draw_menu(self.screen, self.fonts, self.level_options, self.level_selected,
                         title="SELECT LEVEL")
        else:
            self.screen.fill(theme.BG_TOP)
        ui.draw_fade(self.screen, self.fade_alpha)

    def _run_human_play_session(self):
        run_human_play(self.screen, self.clock, self.fonts, self.selected_level_path,
                        exit_to_menu=True)
        self.state = AppState.MENU

    def _run_watch_agent_session(self):
        checkpoint = self.default_checkpoint or find_latest_checkpoint()
        if not checkpoint or not Path(checkpoint).exists():
            self._show_message(
                "No trained checkpoint found",
                "train one first: python -m agent.train --config configs/ppo_baseline.yaml",
            )
            self.state = AppState.MENU
            return
        from stable_baselines3 import PPO

        model = PPO.load(checkpoint)
        run_watch_agent(model, self.selected_level_path, episodes=5,
                         max_episode_steps=MAX_EPISODE_STEPS, seed=0,
                         screen=self.screen, clock=self.clock, fonts=self.fonts,
                         exit_to_menu=True)
        self.state = AppState.MENU

    def _show_message(self, title: str, subtitle: str = "", duration: float = 2.0):
        """Blocking on-screen banner -- used when an action can't proceed (e.g. no checkpoint),
        so the screen never just silently flickers back to the menu with no explanation."""
        elapsed = 0.0
        while elapsed < duration and self.running:
            dt = self.clock.tick(FPS) / 1000.0
            elapsed += dt
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    elapsed = duration  # any key dismisses early
            self.screen.fill(theme.BG_TOP)
            ui.draw_end_banner(self.screen, self.fonts, title, theme.HAZARD_COLOR)
            if subtitle:
                _, font_med, _ = self.fonts
                sub_surf = font_med.render(subtitle, True, theme.TEXT_DIM)
                screen_w, screen_h = self.screen.get_size()
                self.screen.blit(sub_surf, (screen_w / 2 - sub_surf.get_width() / 2, screen_h * 0.5))
            pygame.display.flip()


def run_human_play(screen: pygame.Surface, clock: pygame.time.Clock, fonts, level_path: str,
                    exit_to_menu: bool = False):
    tilemap = TileMap.from_json(level_path)
    world = World(tilemap)
    state = world.reset(seed=np.random.randint(0, 2**31 - 1))
    renderer = Renderer(screen, tilemap)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        action = _keys_to_action(keys)
        prev_on_ground = world.player.on_ground
        state, reward_components, terminated, info = world.tick(action)

        renderer.update(state, dt)
        if terminated and not info.get("goal_reached"):
            renderer.trigger_death_shake()

        renderer.draw(state)
        ui.draw_hud(screen, fonts, "Manual", info["x_progress_frac"],
                    state.elapsed_seconds, MAX_EPISODE_STEPS * DT)
        if terminated:
            banner = "GOAL!" if info.get("goal_reached") else "you died"
            color = theme.GOAL_COLOR if info.get("goal_reached") else theme.HAZARD_COLOR
            ui.draw_end_banner(screen, fonts, banner, color)
        pygame.display.flip()

        if terminated:
            pygame.time.wait(900)
            state = world.reset(seed=np.random.randint(0, 2**31 - 1))

        if exit_to_menu and not running:
            break


def run_watch_agent(model, level_path: str, episodes: int, max_episode_steps: int, seed: int,
                     screen: pygame.Surface | None = None, clock: pygame.time.Clock | None = None,
                     fonts=None, exit_to_menu: bool = False):
    own_display = screen is None
    if own_display:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Mario RL -- watch agent")
        clock = pygame.time.Clock()
        fonts = _make_fonts()

    tilemap = TileMap.from_json(level_path)
    renderer = Renderer(screen, tilemap)

    from env.mario_env import MarioEnv  # local import: obs-building logic lives in the env wrapper
    obs_env = MarioEnv(level_path=level_path, max_episode_steps=max_episode_steps)
    world = obs_env.world  # render directly from the env's own World, no second instance

    running = True
    for ep in range(episodes):
        if not running:
            break
        ep_seed = seed + ep
        obs, _ = obs_env.reset(seed=ep_seed)

        terminated = truncated = False
        while not (terminated or truncated) and running:
            dt = clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            if not running:
                break

            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            obs_tensor = model.policy.obs_to_tensor(obs[None, :])[0]
            import torch
            with torch.no_grad():
                distribution = model.policy.get_distribution(obs_tensor)
                probs = distribution.distribution.probs.cpu().numpy()[0]
                value = model.policy.predict_values(obs_tensor).cpu().numpy()[0, 0]

            obs, reward, terminated, truncated, info = obs_env.step(action)
            state = world._snapshot()

            renderer.update(state, dt)
            if terminated and not info.get("goal_reached"):
                renderer.trigger_death_shake()
            renderer.draw(state)
            ui.draw_hud(screen, fonts, "Agent", info["x_progress_frac"],
                        state.elapsed_seconds, max_episode_steps * DT)
            ui.draw_policy_overlay(screen, fonts, probs, float(value))
            if terminated or truncated:
                banner = "GOAL!" if info.get("goal_reached") else ("timeout" if truncated else "died")
                color = theme.GOAL_COLOR if info.get("goal_reached") else theme.HAZARD_COLOR
                ui.draw_end_banner(screen, fonts, banner, color)
            pygame.display.flip()

        pygame.time.wait(600)

    if own_display:
        pygame.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint to use for 'watch agent' from the menu "
                             "(default: auto-detect the newest runs/*/best_model.zip)")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Mario RL")
    clock = pygame.time.Clock()

    app = App(screen, clock, default_checkpoint=args.checkpoint)
    print(f"[render.app] watch-agent checkpoint: {app.default_checkpoint or '(none found)'}")
    app.run()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
