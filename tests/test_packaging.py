from __future__ import annotations

import ast
import importlib.metadata
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def load_project() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]


def test_all_extra_covers_every_other_extra():
    project = load_project()
    extras = project["optional-dependencies"]
    expected = {f"{project['name']}[{name}]" for name in extras if name != "all"}

    assert set(extras["all"]) == expected


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_distributions(project: dict) -> set[str]:
    requirements = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        requirements.extend(extra)
    return {_normalize(re.match(r"[A-Za-z0-9._-]+", requirement).group(0)) for requirement in requirements}


def _third_party_imports() -> set[str]:
    """Top-level modules imported anywhere in the package, except the stdlib and xdiff itself."""
    modules = set()
    for path in (ROOT / "xdiff").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return {module for module in modules if module not in sys.stdlib_module_names and module != "xdiff"}


def test_every_imported_package_is_declared():
    # A package that is only installed transitively (numpy came in through xarray) has no
    # tested lower bound and can disappear when the dependency that brings it changes.
    declared = _declared_distributions(load_project())
    distributions_by_module = importlib.metadata.packages_distributions()

    undeclared = {}
    for module in sorted(_third_party_imports()):
        distributions = {_normalize(name) for name in distributions_by_module.get(module, [module])}
        if not distributions & declared:
            undeclared[module] = sorted(distributions)

    assert undeclared == {}, f"imported but not declared in pyproject.toml: {undeclared}"
