"""Plotting subpackage: turn a comparison into a picture of where fields differ.

The public surface is :func:`xdiff.plotting.spec.build_plot_spec`, which produces
a backend-agnostic :class:`~xdiff.plotting.spec.PlotSpec`, plus the renderers under
:mod:`xdiff.plotting.renderers` that consume it. Heavy plotting libraries are
imported lazily inside the renderers, mirroring ``load_xarray`` / the dask runtime.
"""

from __future__ import annotations
