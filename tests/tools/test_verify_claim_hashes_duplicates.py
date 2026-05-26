from __future__ import annotations

import pytest

from scripts.verify_claim_hashes import claim_lines


def test_duplicate_claim_id_raises() -> None:
    text = '| C-AAA | x | y | z |\n| C-AAA | x2 | y2 | z2 |'
    with pytest.raises(ValueError):
        claim_lines(text)
