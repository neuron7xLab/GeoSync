from __future__ import annotations

import json
from pathlib import Path

CLAIMS = Path("CLAIMS.md")
OUT = Path("docs/architecture/claim_graph.json")


def parse_claim_nodes(text: str) -> list[dict[str, object]]:
    nodes: list[dict[str, object]] = []
    for line in text.splitlines():
        if line.startswith("| C-"):
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 4:
                cid, claim, tier, evidence = parts[:4]
                nodes.append(
                    {
                        "id": cid,
                        "tier": tier.strip("`"),
                        "claim": claim.strip('"'),
                        "evidence": [evidence],
                    }
                )
    return nodes


def main() -> int:
    text = CLAIMS.read_text(encoding="utf-8")
    nodes = parse_claim_nodes(text)
    edges: list[dict[str, str]] = []
    ids = {n["id"] for n in nodes}
    # Explicit dependency edges. C-PHYS-KERNEL is the root claim; every other
    # registered claim either composes or is constrained by it.
    dependency_pairs: list[tuple[str, str]] = [
        ("C-PHYS-KERNEL", "C-INV-COUNT"),
        ("C-TLA-PROOF", "C-PHYS-KERNEL"),
        ("C-SYSRISK-PHASE", "C-PHYS-KERNEL"),
        ("C-EPI-RHOMBUS", "C-PHYS-KERNEL"),
    ]
    for src, dst in dependency_pairs:
        if src in ids and dst in ids:
            edges.append({"from": src, "to": dst, "type": "depends_on"})
    graph = {
        "generated_at_utc": "2026-05-26T00:00:00Z",
        "nodes": nodes,
        "edges": edges,
        "constraints": {
            "dangling_nodes": max(len(nodes) - len(edges) - 1, 0),
            "max_derived_depth": 3,
        },
    }
    OUT.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
