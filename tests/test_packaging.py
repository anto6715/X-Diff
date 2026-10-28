from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def load_project() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]


def test_all_extra_covers_every_other_extra():
    project = load_project()
    extras = project["optional-dependencies"]
    expected = {f"{project['name']}[{name}]" for name in extras if name != "all"}

    assert set(extras["all"]) == expected
