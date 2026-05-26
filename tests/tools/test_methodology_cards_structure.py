from __future__ import annotations

from pathlib import Path

import yaml


def test_methodology_cards_minimum_structure() -> None:
    cards_doc = yaml.safe_load(Path('docs/architecture/methodology/METHODOLOGY_CARDS.yaml').read_text(encoding='utf-8'))
    assert isinstance(cards_doc, dict)
    assert 'cards' in cards_doc
    assert cards_doc['cards']

    required = {
        'card_id',
        'title',
        'problem_statement',
        'falsifiable_hypothesis',
        'invariant',
        'io_contract',
        'success_criteria',
        'completion_criteria',
        'failure_disposal',
        'automation',
        'owner',
        'updated_at_utc',
    }
    for card in cards_doc['cards']:
        assert required.issubset(card.keys())
