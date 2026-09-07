from __future__ import annotations
import json, re, time
from pathlib import Path
from urllib.parse import urlparse
import requests

BASE='https://www.okx.com'
OUT=Path('delisted_universe_probe'); OUT.mkdir(exist_ok=True)
S=requests.Session(); S.headers.update({'User-Agent':'GLOBAL-CRYPTO-HUNTER-SURVIVORSHIP-AUDIT/1.0','Accept-Language':'en-US'})


def get_json(path,params=None):
    for i in range(7):
        try:
            r=S.get(BASE+path,params=params,timeout=25)
            if r.status_code==200:
                j=r.json()
                if str(j.get('code'))=='0': return j.get('data') or []
        except Exception: pass
        time.sleep(min(5,.4*2**i))
    return []


def get_text(url):
    for i in range(5):
        try:
            r=S.get(url,timeout=25)
            if r.status_code==200: return r.text
        except Exception: pass
        time.sleep(min(4,.5*2**i))
    return ''


def announcement_pages():
    first=get_json('/api/v5/support/announcements',{'annType':'announcements-delistings','page':'1'})
    if not first: return []
    total=int(first[0].get('totalPage') or 1); out=[]
    for page in range(1,total+1):
        data=first if page==1 else get_json('/api/v5/support/announcements',{'annType':'announcements-delistings','page':str(page)})
        if data: out.extend(data[0].get('details') or [])
        time.sleep(.45)
    return out


def normalize_article_text(html):
    # Retain enough text for instrument extraction; no trading decision uses this parsing.
    x=re.sub(r'<script[\s\S]*?</script>|<style[\s\S]*?</style>',' ',html,flags=re.I)
    x=re.sub(r'<[^>]+>',' ',x)
    x=x.replace('&nbsp;',' ').replace('&amp;','&')
    return re.sub(r'\s+',' ',x)


def candidate_symbols(title,text):
    blob=(title+' '+text).upper()
    if 'PERPETUAL' not in blob and 'PERP' not in blob:
        return []
    # OKX announcements commonly render IDs as ABCUSDT. We validate every candidate
    # against the historical-candle endpoint, so false textual matches cannot enter training.
    bases=set(re.findall(r'\b([A-Z0-9]{2,20})USDT\b',blob))
    bad={'USDT','USD','OKX','API','HTTP','HTTPS'}
    return sorted(b for b in bases if b not in bad)


def history_probe(inst):
    data=get_json('/api/v5/market/history-candles',{'instId':inst,'bar':'4H','limit':'3'})
    if not data: return None
    ts=sorted(int(x[0]) for x in data if x)
    return {'instId':inst,'sample_bars':len(data),'oldest_sample_ts':min(ts),'newest_sample_ts':max(ts)} if ts else None


def main():
    anns=announcement_pages(); print('DELISTING_ANNOUNCEMENTS',len(anns),flush=True)
    candidates={}; perpetual_articles=[]
    for n,a in enumerate(anns,1):
        title=str(a.get('title') or ''); url=str(a.get('url') or '')
        if not url: continue
        html=get_text(url); txt=normalize_article_text(html)
        syms=candidate_symbols(title,txt)
        if syms:
            perpetual_articles.append({'title':title,'url':url,'pTime':a.get('pTime'),'bases':syms})
            for b in syms: candidates.setdefault(b,[]).append(title)
        if n%10==0: print('ARTICLES_SCANNED',n,'CANDIDATE_BASES',len(candidates),flush=True)
        time.sleep(.18)
    verified=[]
    for b,titles in sorted(candidates.items()):
        inst=f'{b}-USDT-SWAP'; p=history_probe(inst)
        if p:
            p['announcement_titles']=titles; verified.append(p); print('HISTORY_AVAILABLE',inst,flush=True)
        time.sleep(.14)
    report={'announcement_count':len(anns),'perpetual_article_count':len(perpetual_articles),'candidate_base_count':len(candidates),'verified_delisted_history_count':len(verified),'perpetual_articles':perpetual_articles,'verified_delisted_history':verified,'purpose':'survivorship-bias audit only; no model authority change'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print('FINAL',json.dumps({k:report[k] for k in ('announcement_count','perpetual_article_count','candidate_base_count','verified_delisted_history_count')},sort_keys=True),flush=True)

if __name__=='__main__': main()
