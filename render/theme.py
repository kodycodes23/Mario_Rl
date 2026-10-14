"""Colors, fonts, and easing helpers -- the "slick UI" visual language.

A small, intentional flat/minimal palette (5 colors + neutrals) instead of per-sprite art
assets: every shape in render/renderer.py is a procedurally drawn rounded-rect / polygon, so
restyling the whole game is a matter of editing this one file.
"""
from __future__ import annotations

import math

# --- Palette (5 colors: bg gradient, ground, player accent, enemy accent, UI accent) ----------
BG_TOP = (24, 28, 43)
BG_BOTTOM = (44, 52, 80)
GROUND = (70, 130, 100)
GROUND_EDGE = (94, 168, 130)
BRICK = (176, 108, 74)
BRICK_EDGE = (206, 140, 100)
HAZARD_COLOR = (230, 80, 90)
GOAL_COLOR = (240, 200, 80)
PLAYER_ACCENT = (240, 240, 250)
PLAYER_SHADOW = (180, 190, 220)
ENEMY_ACCENT = (150, 90, 220)
UI_ACCENT = (100, 200, 255)
UI_BG = (18, 20, 32)
UI_BG_ALPHA = 210
TEXT = (235, 238, 245)
TEXT_DIM = (150, 158, 180)

PARALLAX_LAYERS = [
    {"color": (34, 40, 62), "speed": 0.15, "shape": "hills"},
    {"color": (28, 33, 52), "speed": 0.35, "shape": "hills"},
    {"color": (22, 26, 42), "speed": 0.6, "shape": "trees"},
]

FONT_NAME = None  # None -> pygame default font
FONT_SIZE_SMALL = 14
FONT_SIZE_MED = 20
FONT_SIZE_LARGE = 36

RADIUS_SMALL = 4
RADIUS_MED = 8


# --- Easing / interpolation ---------------------------------------------------------------
def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def ease_out_cubic(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return 1 - (1 - t) ** 3


def ease_out_back(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def ease_in_out_sine(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return -(math.cos(math.pi * t) - 1) / 2


def gradient_color(top: tuple, bottom: tuple, t: float) -> tuple:
    t = clamp(t, 0.0, 1.0)
    return tuple(int(lerp(top[i], bottom[i], t)) for i in range(3))
