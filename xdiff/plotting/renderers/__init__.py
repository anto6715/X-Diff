"""Renderers that consume a :class:`~xdiff.plotting.spec.PlotSpec`.

Two lifecycles: ``matplotlib_renderer.render_to_files`` writes static images one-shot;
the interactive Panel/Bokeh server arrives in a later iteration. Heavy imports live
lazily inside the renderer functions, never at module top.
"""

from __future__ import annotations

# Default diverging colormap for the difference (centred at 0).
DIFF_CMAP = "RdBu_r"
# NaN cells (land / masked) are painted this neutral grey. On the diverging colormap white
# already means "no difference", so leaving NaN white would be ambiguous.
LAND_COLOR = "#b0b0b0"
