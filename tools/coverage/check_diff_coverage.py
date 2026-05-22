#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys, tomllib, xml.etree.ElementTree as ET
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--coverage',type=Path,required=True); p.add_argument('--targets',type=Path,required=True); p.add_argument('--base',default='origin/main'); p.add_argument('--head',default='HEAD'); p.add_argument('--min-diff',type=float,default=90); a=p.parse_args()
 if not a.coverage.exists(): print('coverage report missing',file=sys.stderr); return 2
 cfg=tomllib.loads(a.targets.read_text()); critical={}
 for s,d in cfg['surfaces'].items():
  for pref in d.get('paths',[]): critical[pref]=s
 diff=subprocess.check_output(['git','diff','--name-only',a.base,a.head],text=True).splitlines()
 touched=[f for f in diff if any(f.startswith(p) for p in critical)]
 if not touched: print('No critical surface changes detected.'); return 0
 cov={} ; root=ET.parse(a.coverage).getroot()
 for cls in root.findall('.//class'): cov[cls.get('filename','')]=float(cls.get('line-rate','0'))*100
 bad=[]
 for f in touched:
  rate=cov.get(f,0.0)
  if rate<a.min_diff: bad.append((f,rate))
 if bad:
  print('Diff coverage gate failed:'); [print(f'- {f}: {r:.2f}% < {a.min_diff:.2f}%') for f,r in bad]; return 1
 print('Diff coverage gate passed.'); return 0
if __name__=='__main__': raise SystemExit(main())
