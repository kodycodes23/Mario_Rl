"""Menus, HUD, and the watch-agent policy overlay -- the UI chrome layer, pygame only."""
from __future__ import annotations

import pygame

from game.constants import Action
from render import theme

ACTION_NAMES = ["noop", "left", "right", "jump", "left+jump", "right+jump"]


def _panel(screen: pygame.Surface, rect: pygame.Rect, radius: int = theme.RADIUS_MED):
    s = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(s, (*theme.UI_BG, theme.UI_BG_ALPHA), s.get_rect(), border_radius=radius)
    screen.blit(s, rect.topleft)


def draw_hud(screen: pygame.Surface, fonts, mode: str, x_progress_frac: float,
             elapsed_seconds: float, max_episode_seconds: float):
    screen_w, _ = screen.get_size()
    font_small, font_med, _ = fonts

    bar_rect = pygame.Rect(20, 20, 260, 22)
    _panel(screen, bar_rect.inflate(8, 8))
    pygame.draw.rect(screen, (60, 66, 90), bar_rect, border_radius=theme.RADIUS_SMALL)
    fill_w = int(bar_rect.width * theme.clamp(x_progress_frac, 0.0, 1.0))
    if fill_w > 0:
        pygame.draw.rect(screen, theme.UI_ACCENT, (bar_rect.x, bar_rect.y, fill_w, bar_rect.height),
                          border_radius=theme.RADIUS_SMALL)
    label = font_small.render(f"progress {int(x_progress_frac * 100)}%", True, theme.TEXT)
    screen.blit(label, (bar_rect.x + 6, bar_rect.y + 3))

    time_frac = theme.clamp(1.0 - elapsed_seconds / max_episode_seconds, 0.0, 1.0)
    time_text = font_small.render(f"time left {elapsed_seconds:0.1f}s / {max_episode_seconds:0.0f}s",
                                   True, theme.TEXT_DIM)
    screen.blit(time_text, (20, 52))

    mode_rect = pygame.Rect(screen_w - 180, 20, 160, 34)
    _panel(screen, mode_rect)
    mode_text = font_med.render(mode, True, theme.UI_ACCENT)
    screen.blit(mode_text, (mode_rect.x + 12, mode_rect.y + 6))


def draw_menu(screen: pygame.Surface, fonts, options: list[str], selected: int, title: str = "MARIO RL"):
    screen_w, screen_h = screen.get_size()
    _, font_med, font_large = fonts
    screen.fill(theme.BG_TOP)
    for y in range(0, screen_h, 4):
        t = y / screen_h
        pygame.draw.rect(screen, theme.gradient_color(theme.BG_TOP, theme.BG_BOTTOM, t),
                          (0, y, screen_w, 4))

    title_surf = font_large.render(title, True, theme.TEXT)
    screen.blit(title_surf, (screen_w / 2 - title_surf.get_width() / 2, screen_h * 0.22))

    for i, opt in enumerate(options):
        y = screen_h * 0.45 + i * 50
        rect = pygame.Rect(0, 0, 320, 40)
        rect.center = (screen_w / 2, y)
        color = theme.UI_ACCENT if i == selected else theme.UI_BG
        pygame.draw.rect(screen, color, rect, border_radius=theme.RADIUS_MED)
        text_color = (20, 24, 36) if i == selected else theme.TEXT
        text = font_med.render(opt, True, text_color)
        screen.blit(text, (rect.centerx - text.get_width() / 2, rect.centery - text.get_height() / 2))

    hint = font_med.render("up/down to select, enter to confirm, esc to quit", True, theme.TEXT_DIM)
    screen.blit(hint, (screen_w / 2 - hint.get_width() / 2, screen_h * 0.85))


def draw_policy_overlay(screen: pygame.Surface, fonts, action_probs, value_estimate: float):
    """Live bar chart of action probabilities + value estimate, for watch-agent mode.

    Makes the policy's decision legible rather than a black box (research-communication touch).
    """
    font_small, font_med, _ = fonts
    screen_w, screen_h = screen.get_size()
    panel_w, panel_h = 220, 30 + 22 * len(action_probs) + 30
    rect = pygame.Rect(screen_w - panel_w - 20, screen_h - panel_h - 20, panel_w, panel_h)
    _panel(screen, rect)

    title = font_small.render("policy (action probs)", True, theme.TEXT_DIM)
    screen.blit(title, (rect.x + 10, rect.y + 8))

    bar_max_w = panel_w - 100
    for i, p in enumerate(action_probs):
        y = rect.y + 30 + i * 22
        name = font_small.render(ACTION_NAMES[i], True, theme.TEXT)
        screen.blit(name, (rect.x + 10, y))
        bar_rect = pygame.Rect(rect.x + 80, y + 2, int(bar_max_w * p), 12)
        pygame.draw.rect(screen, theme.UI_ACCENT, bar_rect, border_radius=3)

    value_text = font_small.render(f"V(s) ~ {value_estimate:0.2f}", True, theme.TEXT_DIM)
    screen.blit(value_text, (rect.x + 10, rect.bottom - 24))


def draw_fade(screen: pygame.Surface, alpha: float):
    if alpha <= 0:
        return
    s = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    s.fill((10, 10, 16, int(255 * theme.clamp(alpha, 0.0, 1.0))))
    screen.blit(s, (0, 0))


def draw_end_banner(screen: pygame.Surface, fonts, text: str, color=theme.TEXT):
    _, font_med, font_large = fonts
    screen_w, screen_h = screen.get_size()
    surf = font_large.render(text, True, color)
    rect = surf.get_rect(center=(screen_w / 2, screen_h * 0.35))
    _panel(screen, rect.inflate(40, 24))
    screen.blit(surf, rect)
