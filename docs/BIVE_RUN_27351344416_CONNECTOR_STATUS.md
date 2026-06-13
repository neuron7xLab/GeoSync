# BIVE run 27351344416 connector integration evidence

Status: PARTIAL.

Connector facts:
- Repository search confirms neuron7xLab/bive exists, is public, default branch is main, and the connector reports admin/push permissions.
- A documentation mutation was accepted by GitHub, proving the write path can reach neuron7xLab/bive.
- Code and packaging mutations for Makefile, pyproject.toml, and src package paths were blocked before GitHub accepted them.

Repository mismatch observed through connector reads:
- README and Makefile content describe GeoSync.
- pyproject metadata also resolves to GeoSync on fetch_file.
- src/geosync exists.
- src/bive/cli.py is missing.
- scripts/verify_release.py is missing.
- .github/workflows/ci.yml is missing.

Root cause class:
BIVE repository identity and package body are inconsistent. The installed-package proof path cannot be made green until the repository contains a real bive import package, console entrypoints, release scripts, dependency constraints, and CI workflow.

No release green status is claimed.
