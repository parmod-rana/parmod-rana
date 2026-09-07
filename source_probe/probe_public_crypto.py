import json, requests
urls={
 'bybit':'https://api.bybit.com/v5/market/time',
 'okx':'https://www.okx.com/api/v5/public/time',
 'kraken':'https://api.kraken.com/0/public/Time',
 'coinbase':'https://api.exchange.coinbase.com/time'
}
for name,url in urls.items():
    try:
        r=requests.get(url,timeout=15,headers={'User-Agent':'GLOBAL-CRYPTO-HUNTER/1.0'})
        print(name, r.status_code, r.text[:300].replace('\n',' '))
    except Exception as e:
        print(name,'ERROR',repr(e))
