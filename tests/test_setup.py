import pytest

import xdiff


def test_setup_configures_logging_for_supported_python(monkeypatch):
    called = []

    monkeypatch.setattr(xdiff.sys, "version_info", (3, 13, 0, "final", 0))
    monkeypatch.setattr(
        xdiff,
        "configure_logging",
        lambda config, logging: called.append((config, logging)),
    )

    xdiff.setup()

    assert called == [(xdiff.settings.LOGGING_CONFIG, xdiff.settings.LOGGING)]


def test_setup_rejects_unsupported_python_before_configuring_logging(monkeypatch):
    called = []

    monkeypatch.setattr(xdiff.sys, "version_info", (3, 14, 0, "final", 0))
    monkeypatch.setattr(
        xdiff,
        "configure_logging",
        lambda config, logging: called.append((config, logging)),
    )

    with pytest.raises(RuntimeError, match="supports Python 3.10 through 3.13"):
        xdiff.setup()

    assert called == []
