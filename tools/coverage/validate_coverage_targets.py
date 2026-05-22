#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, tomllib
from pathlib import Path

def main():
  p=argparse.ArgumentParser(); p.add_argument('--coverage',type=Path,required=True); p.add_argument('--targets',type=Path,required=True); a=p.parse_args()
  if not a.coverage.exists(): print('missing coverage xml',file=sys.stderr); return 2
  for needed in ['coverage_summary.json','module_coverage.json']:
    if not (Path('reports/coverage')/needed).exists(): print(f'missing reports/coverage/{needed}',file=sys.stderr); return 1
  cfg=tomllib.loads(a.targets.read_text()); summary=json.loads((Path('reports/coverage')/'coverage_summary.json').read_text())
  critical=['core','execution','backtest','risk']
  for c in critical:
    if c not in summary.get('surfaces',{}): print(f'missing surface {c}',file=sys.stderr); return 1
    if summary['surfaces'][c]['line_coverage']<float(cfg['surfaces'][c]['short_term']): print(f'{c} below threshold',file=sys.stderr); return 1
  pm=list(Path('reports/coverage').glob('priority_map_*.md'))
  if not pm: print('priority_map missing',file=sys.stderr); return 1
  print('coverage targets validated'); return 0
if __name__=='__main__': raise SystemExit(main())
