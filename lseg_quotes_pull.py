"""Pull LSEG end-of-day BID/ASK for the bundle RICs (prereg/spread_check_v1.json).
Needs LSEG Workspace running and signed in, and LSEG_APP_KEY in the environment (never in this file).
usage: python lseg_quotes_pull.py STATIC_CSV OUT_CSV  (resumable)"""
import sys
import lseg.data as ld, pandas as pd, os
ld.open_session(app_key=os.environ['LSEG_APP_KEY'])
rics = pd.read_csv(sys.argv[1])['Instrument'].dropna().unique().tolist()
out = sys.argv[2]
done = set(pd.read_csv(out, usecols=['ric'])['ric']) if os.path.exists(out) else set()
todo = [r for r in rics if r not in done]
print(len(rics), 'rics,', len(todo), 'to do')
fails = []
for i, r in enumerate(todo):
    d, src = None, 'history'
    try:
        h = ld.get_history(r, fields=['BID', 'ASK'], interval='daily', start='2015-09-01', end='2026-10-30')
        h.columns = [str(c).lower() for c in h.columns]
        if len(h) and {'bid', 'ask'} <= set(h.columns):
            d = h[['bid', 'ask']].reset_index()
            d.columns = ['date', 'bid', 'ask']
    except Exception as e:
        pass
    if d is None or d[['bid', 'ask']].notna().all(axis=1).sum() == 0:
        src = 'tr_fields'
        try:
            g = ld.get_data(r, ['TR.BIDPRICE.date', 'TR.BIDPRICE', 'TR.ASKPRICE'],
                            {'SDate': '2015-09-01', 'EDate': '2026-10-30', 'Frq': 'D'})
            d = g.iloc[:, 1:4]
            d.columns = ['date', 'bid', 'ask']
        except Exception as e:
            d = None
    if d is None or len(d) == 0:
        fails.append(r)
        d = pd.DataFrame({'date': [None], 'bid': [None], 'ask': [None]})
        src = 'none'
    d.insert(0, 'ric', r)
    d['src'] = src
    d.to_csv(out, mode='a', header=not os.path.exists(out), index=False)
    if i % 50 == 0:
        print(i, r, src, len(d))
print('fails', len(fails), fails[:20])
print('DONE')
