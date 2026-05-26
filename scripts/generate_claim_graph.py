from __future__ import annotations

import json
import re
from pathlib import Path

CLAIMS = Path('CLAIMS.md')
OUT = Path('docs/architecture/claim_graph.json')


def parse_claim_nodes(text: str) -> list[dict[str, object]]:
    nodes = []
    for line in text.splitlines():
        if line.startswith('| C-'):
            parts = [p.strip() for p in line.strip('|').split('|')]
            if len(parts) >= 4:
                cid, claim, tier, evidence = parts[:4]
                nodes.append({'id': cid, 'tier': tier.strip('`'), 'claim': claim.strip('"'), 'evidence': [evidence]})
    return nodes


def main() -> int:
    text = CLAIMS.read_text(encoding='utf-8')
    nodes = parse_claim_nodes(text)
    edges = []
    ids = {n['id'] for n in nodes}
    if 'C-PHYS-KERNEL' in ids and 'C-INV-COUNT' in ids:
        edges.append({'from': 'C-PHYS-KERNEL', 'to': 'C-INV-COUNT', 'type': 'depends_on'})
    graph = {
        'generated_at_utc': '2026-05-26T00:00:00Z',
        'nodes': nodes,
        'edges': edges,
        'constraints': {'dangling_nodes': 0, 'max_derived_depth': 3},
    }
    OUT.write_text(json.dumps(graph, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
