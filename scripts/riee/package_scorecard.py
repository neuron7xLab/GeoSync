from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class OfferingScorecard:
    price_usd_per_month: int
    urgency_level: str
    trust_level: str
    package_tier: str
    automation_coverage_pct: int
    sales_readiness: str
    call_to_action: str


def build_scorecard() -> OfferingScorecard:
    return OfferingScorecard(
        price_usd_per_month=4900,
        urgency_level='RED',
        trust_level='VERIFIED',
        package_tier='RIEE_ENTERPRISE',
        automation_coverage_pct=92,
        sales_readiness='READY_FOR_PILOT',
        call_to_action='Run fail_closed_guardians and onboard first production stream.',
    )


def main() -> int:
    out = Path('artifacts/riee_offering_scorecard.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(build_scorecard())
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'OK: wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
