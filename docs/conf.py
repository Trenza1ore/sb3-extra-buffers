"""Sphinx configuration for sb3-extra-buffers."""

# pylint: disable=redefined-builtin

import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

project = "sb3-extra-buffers"
author = "Hugo Huang"
copyright = "2026, Hugo Huang"

_version_file = PROJECT_ROOT / "sb3_extra_buffers" / "version.txt"
if _version_file.is_file():
    version = release = _version_file.read_text(encoding="utf-8").strip()
else:
    version = release = "0.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "myst_parser",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autoclass_content = "both"
add_module_names = False

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_mock_imports = ["ale_py"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
    "stable_baselines3": ("https://stable-baselines3.readthedocs.io/en/master/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"{project} documentation"

html_theme_options = {
    "style_external_links": True,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "prev_next_buttons_location": "both",
}

pygments_style = "friendly"
language = "en"
nitpicky = False
