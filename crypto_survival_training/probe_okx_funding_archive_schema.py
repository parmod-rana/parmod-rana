from __future__ import annotations
import csv, io, json, zipfile, requests
from pathlib import Path

OUT=Path('funding_archive_schema'); OUT.mkdir(exist_ok=True)
URL='https://static.okx.com/cdn/okex/traderecords/swaprates/daily/20250105/allswap-fundingrates-2025-01-05.zip?v=999'

def main():
    r=requests.get(URL,timeout=60); r.raise_for_status()
    raw=r.content
    z=zipfile.ZipFile(io.BytesIO(raw))
    names=z.namelist()
    report={'url':URL,'zip_bytes':len(raw),'files':names,'members':[]}
    for name in names[:5]:
        data=z.read(name)
        text=data.decode('utf-8-sig',errors='replace')
        lines=text.splitlines()
        sample=lines[:8]
        report['members'].append({'name':name,'bytes':len(data),'line_count':len(lines),'sample_lines':sample})
    print(json.dumps(report,indent=2)); (OUT/'schema.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__': main()
