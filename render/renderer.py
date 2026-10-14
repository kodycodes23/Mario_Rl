"""Pygame renderer: reads World state and draws it. Writes nothing back into World/MarioEnv.

Every sprite is procedurally drawn (rounded rects + polygons) rather than loaded art, per the
"slick, small, intentional" visual language in theme.py. This module is the only place pygame
matters -- see mario_rl_build_prompt.md §7's performance boundary: env/ and agent/train.py must
never import this.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import pygame

from game.constants import Tile
from game.world import WorldState
from render import theme

PX_PER_TILE = 32


def tile_to_px(v: float) -> float:
    return v * PX_PER_TILE


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: tuple
    radius: float = 3.0

    def step(self, dt: float) -> bool:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 20.0 * dt
        self.life -= dt
        return self.life > 0

    def draw(self, surface: pygame.Surface, cam_x: float, cam_y: float):
        t = max(self.life / self.max_life, 0.0)
        alpha = int(255 * t)
        r = max(1.0, self.radius * t)
        color = (*self.color, alpha)
        s = pygame.Surface((int(r * 2) + 2, int(r * 2) + 2), pygame.SRCALPHA)
        pygame.draw.circle(s, color, (r + 1, r + 1), r)
        surface.blit(s, (self.x - cam_x - r, self.y - cam_y - r))


class ParticleSystem:
    def __init__(self):
        self.particles: list[Particle] = []

    def emit_puff(self, x: float, y: float, color: tuple, n: int = 8):
        for _ in range(n):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(20, 80)
            self.particles.append(Particle(
                x=x, y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed - 40,
                life=random.uniform(0.25, 0.5),
                max_life=0.5,
                color=color,
                radius=random.uniform(2, 4),
            ))

    def step(self, dt: float):
        self.particles = [p for p in self.particles if p.step(dt)]

    def draw(self, surface: pygame.Surface, cam_x: float, cam_y: float):
        for p in self.particles:
            p.draw(surface, cam_x, cam_y)


class Camera:
    """Smooth follow-camera with lerp easing, not a hard lock."""

    def __init__(self, screen_w: int, screen_h: int, level_w_px: float, level_h_px: float):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.level_w_px = level_w_px
        self.level_h_px = level_h_px
        self.x = 0.0
        self.y = 0.0
        self.shake_time = 0.0
        self.shake_mag = 0.0

    def update(self, target_x_px: float, target_y_px: float, dt: float, lerp_speed: float = 6.0):
        desired_x = target_x_px - self.screen_w / 2
        desired_y = target_y_px - self.screen_h / 2
        t = 1.0 - math.exp(-lerp_speed * dt)
        self.x = theme.lerp(self.x, desired_x, t)
        self.y = theme.lerp(self.y, desired_y, t)

        max_x = max(0.0, self.level_w_px - self.screen_w)
        self.x = theme.clamp(self.x, 0.0, max_x)
        self.y = theme.clamp(self.y, -200, max(0.0, self.level_h_px - self.screen_h) + 200)

        if self.shake_time > 0:
            self.shake_time -= dt

    def trigger_shake(self, magnitude: float = 6.0, duration: float = 0.25):
        self.shake_mag = magnitude
        self.shake_time = duration

    def offset(self) -> tuple[float, float]:
        if self.shake_time > 0:
            f = self.shake_time / 0.25
            return (
                self.x + random.uniform(-1, 1) * self.shake_mag * f,
                self.y + random.uniform(-1, 1) * self.shake_mag * f,
            )
        return self.x, self.y


class PlayerAnim:
    """Tracks squash/stretch + limb-wobble animation state, purely visual (never fed back)."""

    def __init__(self):
        self.squash_timer = 0.0
        self.wobble_phase = 0.0

    def on_landed(self):
        self.squash_timer = 0.18

    def step(self, dt: float, moving: bool):
        if self.squash_timer > 0:
            self.squash_timer = max(0.0, self.squash_timer - dt)
        if moving:
            self.wobble_phase += dt * 14.0

    def scale(self) -> tuple[float, float]:
        if self.squash_timer <= 0:
            return 1.0, 1.0
        t = self.squash_timer / 0.18
        squash = theme.ease_out_back(1 - t)
        sx = 1.0 + 0.25 * (1 - squash)
        sy = 1.0 - 0.25 * (1 - squash)
        return sx, sy


class Renderer:
    def __init__(self, screen: pygame.Surface, tilemap):
        self.screen = screen
        self.tilemap = tilemap
        self.screen_w, self.screen_h = screen.get_size()
        level_w_px = tilemap.width * PX_PER_TILE
        level_h_px = tilemap.height * PX_PER_TILE
        self.camera = Camera(self.screen_w, self.screen_h, level_w_px, level_h_px)
        self.particles = ParticleSystem()
        self.player_anim = PlayerAnim()
        self._prev_on_ground = True
        self.font_small = pygame.font.Font(theme.FONT_NAME, theme.FONT_SIZE_SMALL)
        self.font_med = pygame.font.Font(theme.FONT_NAME, theme.FONT_SIZE_MED)
        self.font_large = pygame.font.Font(theme.FONT_NAME, theme.FONT_SIZE_LARGE)
        self._parallax_seed = random.Random(0)
        self._parallax_shapes = self._build_parallax_shapes()

    def _build_parallax_shapes(self):
        shapes = []
        rng = random.Random(42)
        for layer in theme.PARALLAX_LAYERS:
            n = 14
            layer_shapes = []
            for i in range(n):
                x = rng.uniform(0, self.tilemap.width * PX_PER_TILE * 1.5)
                h = rng.uniform(30, 90)
                w = rng.uniform(60, 140)
                layer_shapes.append((x, h, w))
            shapes.append(layer_shapes)
        return shapes

    def update(self, state: WorldState, dt: float):
        player = state.player
        target_x = tile_to_px(player.x) + PX_PER_TILE / 2
        target_y = tile_to_px(player.y) + PX_PER_TILE / 2
        self.camera.update(target_x, target_y, dt)

        if player.just_landed:
            self.player_anim.on_landed()
            self.particles.emit_puff(target_x, tile_to_px(player.y) + PX_PER_TILE,
                                      theme.PLAYER_SHADOW, n=6)
        if player.just_jumped:
            self.particles.emit_puff(target_x, tile_to_px(player.y) + PX_PER_TILE,
                                      theme.PLAYER_SHADOW, n=4)

        moving = abs(player.vx) > 0.5
        self.player_anim.step(dt, moving)
        self.particles.step(dt)

    def trigger_death_shake(self):
        self.camera.trigger_shake(magnitude=8.0, duration=0.3)

    # ------------------------------------------------------------------ draw
    def draw(self, state: WorldState):
        self._draw_background()
        cam_x, cam_y = self.camera.offset()
        self._draw_parallax(cam_x, cam_y)
        self._draw_tilemap(cam_x, cam_y)
        for hazard in state.hazards:
            self._draw_hazard(hazard, cam_x, cam_y)
        self._draw_goal(state.goal, cam_x, cam_y)
        for enemy in state.enemies:
            if enemy.alive:
                self._draw_enemy(enemy, cam_x, cam_y)
        self._draw_player(state.player, cam_x, cam_y)
        self.particles.draw(self.screen, cam_x, cam_y)

    def _draw_background(self):
        for y in range(0, self.screen_h, 4):
            t = y / self.screen_h
            color = theme.gradient_color(theme.BG_TOP, theme.BG_BOTTOM, t)
            pygame.draw.rect(self.screen, color, (0, y, self.screen_w, 4))

    def _draw_parallax(self, cam_x: float, cam_y: float):
        for layer_idx, layer in enumerate(theme.PARALLAX_LAYERS):
            speed = layer["speed"]
            offset_x = cam_x * speed
            for (x, h, w) in self._parallax_shapes[layer_idx]:
                draw_x = x - offset_x
                # wrap horizontally for an endless-feeling backdrop
                draw_x = draw_x % (self.tilemap.width * PX_PER_TILE * 1.5 + w) - w
                rect = pygame.Rect(int(draw_x), self.screen_h - int(h) - 60, int(w), int(h))
                pygame.draw.rect(self.screen, layer["color"], rect,
                                  border_top_left_radius=12, border_top_right_radius=12)

    def _draw_tilemap(self, cam_x: float, cam_y: float):
        first_col = max(0, int(cam_x // PX_PER_TILE) - 1)
        last_col = min(self.tilemap.width, int((cam_x + self.screen_w) // PX_PER_TILE) + 2)
        first_row = max(0, int(cam_y // PX_PER_TILE) - 1)
        last_row = min(self.tilemap.height, int((cam_y + self.screen_h) // PX_PER_TILE) + 2)

        for row in range(first_row, last_row):
            for col in range(first_col, last_col):
                tile_id = self.tilemap.tile_at(col, row)
                px = col * PX_PER_TILE - cam_x
                py = row * PX_PER_TILE - cam_y
                if tile_id == Tile.GROUND:
                    self._draw_block(px, py, theme.GROUND, theme.GROUND_EDGE)
                elif tile_id == Tile.BRICK:
                    self._draw_block(px, py, theme.BRICK, theme.BRICK_EDGE)

    def _draw_block(self, px: float, py: float, color: tuple, edge_color: tuple):
        rect = pygame.Rect(int(px), int(py), PX_PER_TILE, PX_PER_TILE)
        pygame.draw.rect(self.screen, color, rect, border_radius=theme.RADIUS_SMALL)
        top_rect = pygame.Rect(int(px), int(py), PX_PER_TILE, 5)
        pygame.draw.rect(self.screen, edge_color, top_rect,
                          border_top_left_radius=theme.RADIUS_SMALL,
                          border_top_right_radius=theme.RADIUS_SMALL)

    def _draw_hazard(self, hazard, cam_x: float, cam_y: float):
        px = tile_to_px(hazard.x) - cam_x
        py = tile_to_px(hazard.y) - cam_y
        n_spikes = 3
        w = PX_PER_TILE / n_spikes
        for i in range(n_spikes):
            base_x = px + i * w
            points = [(base_x, py + PX_PER_TILE), (base_x + w / 2, py + PX_PER_TILE * 0.3),
                      (base_x + w, py + PX_PER_TILE)]
            pygame.draw.polygon(self.screen, theme.HAZARD_COLOR, points)

    def _draw_goal(self, goal, cam_x: float, cam_y: float):
        if goal is None:
            return
        px = tile_to_px(goal.x) - cam_x
        py = tile_to_px(goal.y) - cam_y
        pole_rect = pygame.Rect(int(px + PX_PER_TILE * 0.4), int(py - PX_PER_TILE), 4,
                                 int(PX_PER_TILE * 2))
        pygame.draw.rect(self.screen, theme.TEXT_DIM, pole_rect)
        flag_points = [
            (px + PX_PER_TILE * 0.44, py - PX_PER_TILE * 0.9),
            (px + PX_PER_TILE * 1.1, py - PX_PER_TILE * 0.65),
            (px + PX_PER_TILE * 0.44, py - PX_PER_TILE * 0.4),
        ]
        pygame.draw.polygon(self.screen, theme.GOAL_COLOR, flag_points)

    def _draw_enemy(self, enemy, cam_x: float, cam_y: float):
        px = tile_to_px(enemy.x) - cam_x
        py = tile_to_px(enemy.y) - cam_y
        w = PX_PER_TILE * 0.8
        h = PX_PER_TILE * 0.8
        rect = pygame.Rect(int(px), int(py), int(w), int(h))
        pygame.draw.rect(self.screen, theme.ENEMY_ACCENT, rect, border_radius=theme.RADIUS_MED)
        eye_y = int(py + h * 0.35)
        pygame.draw.circle(self.screen, theme.TEXT, (int(px + w * 0.3), eye_y), 3)
        pygame.draw.circle(self.screen, theme.TEXT, (int(px + w * 0.7), eye_y), 3)

    def _draw_player(self, player, cam_x: float, cam_y: float):
        w = PX_PER_TILE * 0.8
        h = PX_PER_TILE * 0.95
        sx, sy = self.player_anim.scale()
        draw_w, draw_h = w * sx, h * sy
        cx = tile_to_px(player.x) - cam_x + w / 2
        cy = tile_to_px(player.y) - cam_y + h
        rect = pygame.Rect(int(cx - draw_w / 2), int(cy - draw_h), int(draw_w), int(draw_h))
        pygame.draw.rect(self.screen, theme.PLAYER_ACCENT, rect, border_radius=theme.RADIUS_MED)

        # simple limb wobble: two little "feet" that swap phase while moving
        if abs(player.vx) > 0.5:
            phase = math.sin(self.player_anim.wobble_phase)
        else:
            phase = 0.0
        foot_w, foot_h = draw_w * 0.35, 5
        foot_y = cy - foot_h / 2
        offset = phase * draw_w * 0.15
        pygame.draw.rect(self.screen, theme.PLAYER_SHADOW,
                          (int(cx - draw_w / 2 + offset), int(foot_y), int(foot_w), int(foot_h)),
                          border_radius=2)
        pygame.draw.rect(self.screen, theme.PLAYER_SHADOW,
                          (int(cx + draw_w / 2 - foot_w - offset), int(foot_y), int(foot_w), int(foot_h)),
                          border_radius=2)

        # facing indicator: a small eye offset toward facing direction
        eye_x = cx + (draw_w * 0.2 if player.facing_right else -draw_w * 0.2)
        eye_y = cy - draw_h * 0.7
        pygame.draw.circle(self.screen, (30, 30, 40), (int(eye_x), int(eye_y)), 3)
