# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""GeoSync command-line entry points (``amm_cli``, ``geosync_cli``).

This file makes ``cli`` a regular package instead of an implicit namespace
package. Without it, ``cli`` is a namespace package that a same-named regular
module (``scripts/cli.py`` -> ``scripts.cli``) can shadow on sys.path, so
``from cli import amm_cli`` resolved to ``scripts/cli.py`` under the CI editable
install and raised ImportError. With ROOT inserted at ``sys.path[0]`` (root
``conftest.py``), a regular ``cli`` package is found first and authoritatively.
"""
