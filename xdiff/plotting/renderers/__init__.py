"""Renderers that consume a :class:`~xdiff.plotting.spec.PlotSpec`.

Two lifecycles: ``matplotlib_renderer.render_to_files`` writes static images one-shot;
the interactive Panel/Bokeh server arrives in a later iteration. Heavy imports live
lazily inside the renderer functions, never at module top.
"""

from __future__ import annotations
