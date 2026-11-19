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
    "canvas": "#242649",
    "card": "#222B68",
    "raised": "#23337C",
    "deep_sapphire": "#263B84",
    "border": "#364989",
    "primary": "#386CAF",
    "accent": "#65A7D3",
    "text_muted": "#B2D3EB",
    "text_emp": "#CDDFEF",
    "text_light": "#FFFFFF",
}

COLOR_RAMP = ["#242649", "#263B84", "#386CAF", "#65A7D3", "#CDDFEF"]


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
