"""PurifEye Design System - Plot Theme and Color Tokens.

Strict blue-palette architecture:
Base Canvas: #242649
Cards & Panels: #222B68
Raised Surfaces: #23337C
Deep Sapphire: #263B84
Hairlines & Borders: #364989
Primary Action: #386CAF
Accent & Bounding Box: #65A7D3
Secondary Text / Muted: #B2D3EB
Emphasized Secondary: #CDDFEF
Primary Text: #FFFFFF
"""

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib as mpl
import matplotlib.pyplot as plt

PURIFEYE_COLORS = {
    # Deep Blue Surfaces
    "canvas": "#060e1f",
    "card": "#0a1836",
    "raised": "#0f224b",
    "deep_sapphire": "#152e63",
    "border": "#1b3975",
    # Ocean Blue Accents
    "primary": "#0284c7",
    "accent": "#38bdf8",
    "ocean_marine": "#00b4d8",
    "text_muted": "#8ac5e8",
    "text_emp": "#d0e8f7",
    "text_light": "#FFFFFF",
}

COLOR_RAMP = ["#060e1f", "#152e63", "#0284c7", "#38bdf8", "#d0e8f7"]


def apply_purifeye_theme():
    """Apply the PurifEye high-contrast dark blue theme to Matplotlib rcParams."""
    plt.style.use("default")
    mpl.rcParams["figure.facecolor"] = PURIFEYE_COLORS["canvas"]
    mpl.rcParams["axes.facecolor"] = PURIFEYE_COLORS["card"]
    mpl.rcParams["savefig.facecolor"] = PURIFEYE_COLORS["canvas"]
    mpl.rcParams["axes.edgecolor"] = PURIFEYE_COLORS["border"]
    mpl.rcParams["axes.labelcolor"] = PURIFEYE_COLORS["text_emp"]
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.color"] = PURIFEYE_COLORS["border"]
    mpl.rcParams["grid.linestyle"] = "--"
    mpl.rcParams["grid.alpha"] = 0.6
    mpl.rcParams["xtick.color"] = PURIFEYE_COLORS["text_muted"]
    mpl.rcParams["ytick.color"] = PURIFEYE_COLORS["text_muted"]
    mpl.rcParams["text.color"] = PURIFEYE_COLORS["text_light"]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Plus Jakarta Sans", "DejaVu Sans", "Arial"]
    mpl.rcParams["lines.color"] = PURIFEYE_COLORS["accent"]
    mpl.rcParams["lines.linewidth"] = 2.0
