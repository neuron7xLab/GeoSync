#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, subprocess, tomllib, xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

@dataclass
class FileCov:
    filename:str; line:float; branch:float; missed:int

def load_cov(xml_path:Path):
    root=ET.parse(xml_path).getroot(); files=[]
    for cls in root.findall('.//class'):
        fn=cls.get('filename','')
        lr=float(cls.get('line-rate','0'))*100
        br=float(cls.get('branch-rate','0'))*100 if cls.get('branch-rate') is not None else 0.0
        lines=cls.findall('./lines/line')
        missed=sum(1 for l in lines if int(l.get('hits','0'))==0)
        files.append(FileCov(fn,lr,br,missed))
    return root,files

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--coverage',type=Path,required=True); ap.add_argument('--targets',type=Path,required=True); ap.add_argument('--out-dir',type=Path,default=Path('reports/coverage'))
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    root,files=load_cov(a.coverage); cfg=tomllib.loads(a.targets.read_text())
    surfaces=cfg.get('surfaces',{})
    module={}
    for name,data in surfaces.items():
        paths=data.get('paths',[]); m=[f for f in files if any(f.filename.startswith(p) for p in paths)]
        line=sum(f.line for f in m)/len(m) if m else 0.0; branch=sum(f.branch for f in m)/len(m) if m else 0.0
        module[name]={"line_coverage":round(line,2),"branch_coverage":round(branch,2),"threshold":data['short_term'],"status":"pass" if line>=float(data['short_term']) else "fail","claim_risk":data['claim_risk'],"files":len(m)}
    mod_path=a.out_dir/'module_coverage.json'; mod_path.write_text(json.dumps(module,indent=2))
    top=sorted(files,key=lambda f:(f.line,f.branch,-f.missed))[:20]
    pm=a.out_dir/f"priority_map_{datetime.now(timezone.utc).strftime('%Y%m%d')}.md"
    rows=["| file_path | surface | line_coverage | branch_coverage | uncovered_lines | claim_dependencies | risk_score | action |","|---|---:|---:|---:|---:|---|---:|---|"]
    for i,f in enumerate(top,1):
        surf=next((s for s,d in surfaces.items() if any(f.filename.startswith(p) for p in d.get('paths',[]))),"unmapped")
        score=round((100-f.line)+(100-f.branch)+min(f.missed,20)+(30 if surf in {'execution','backtest','risk'} else 10),2)
        rows.append(f"| {f.filename} | {surf} | {f.line:.2f}% | {f.branch:.2f}% | {f.missed} | {surf.upper()}_CLAIMS | {score} | add_tests |")
    pm.write_text("\n".join(["# Coverage Priority Map",""]+rows)+"\n")
    summary={"git_sha":subprocess.getoutput('git rev-parse HEAD'),"timestamp":datetime.now(timezone.utc).isoformat(),"global":{"line_coverage":round(float(root.get('line-rate','0'))*100,2),"branch_coverage":round(float(root.get('branch-rate','0'))*100,2),"threshold":cfg['global']['current_release_gate'],"status":"pass" if float(root.get('line-rate','0'))*100>=cfg['global']['current_release_gate'] else "fail"},"surfaces":module,"artifacts":{"coverage_xml":str(a.coverage),"htmlcov":"reports/coverage/htmlcov/","priority_map":str(pm)}}
    (a.out_dir/'coverage_summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__': main()
