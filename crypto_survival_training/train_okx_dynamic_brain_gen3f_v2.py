from __future__ import annotations
import io, hashlib, time, zipfile

import pandas as pd
import requests

from crypto_survival_training import train_okx_dynamic_brain_gen3f as base

# GEN3F V2 DATA-INTEGRITY PATCH ONLY.
# Trading architecture, neural model, folds, costs, q-floor, capacity and all
# frozen promotion gates remain exactly those defined in Gen3F.
# OKX funding archives have been observed with epoch-ms values; this parser also
# accepts genuine datetime text if the upstream schema changes. Never infer a
# timestamp unit from pandas datetime storage resolution.

MIN_FUNDING_ROWS = 50_000
MIN_FUNDING_SYMBOLS = 40
EARLY_COVERAGE_MS = int(pd.Timestamp('2024-01-01', tz='UTC').timestamp() * 1000)
LATE_COVERAGE_MS = int(pd.Timestamp('2026-06-01', tz='UTC').timestamp() * 1000)
MIN_VALID_EPOCH_MS = 1_600_000_000_000
MAX_VALID_EPOCH_MS = 2_000_000_000_000


def _funding_time_to_epoch_ms(values: pd.Series) -> pd.Series:
    """Parse mixed OKX funding_time values without guessing numeric units."""
    raw = values.astype('string').str.strip()
    numeric = pd.to_numeric(raw, errors='coerce')
    out = pd.Series(pd.NA, index=values.index, dtype='Int64')

    # Verified OKX archive representation: 13-digit Unix epoch milliseconds.
    valid_num = numeric.between(MIN_VALID_EPOCH_MS, MAX_VALID_EPOCH_MS, inclusive='both')
    if valid_num.any():
        out.loc[valid_num] = numeric.loc[valid_num].round().astype('int64')

    # Schema-adaptive fallback for genuinely textual UTC datetimes only.
    remaining = out.isna() & raw.notna()
    if remaining.any():
        parsed = pd.to_datetime(raw.loc[remaining], utc=True, errors='coerce')
        text_ms = parsed.map(lambda ts: int(ts.timestamp() * 1000) if pd.notna(ts) else pd.NA).astype('Int64')
        text_valid = text_ms.between(MIN_VALID_EPOCH_MS, MAX_VALID_EPOCH_MS, inclusive='both').fillna(False)
        out.loc[text_ms.index[text_valid]] = text_ms.loc[text_valid]
    return out


def _self_test_timestamp_parser():
    sample = pd.Series(['1735776000000', '2025-01-02 00:00:00+00:00', 'bad'])
    got = _funding_time_to_epoch_ms(sample)
    expected = 1735776000000
    if int(got.iloc[0]) != expected or int(got.iloc[1]) != expected or pd.notna(got.iloc[2]):
        raise RuntimeError(f'funding timestamp parser self-test failed: {got.tolist()}')
    print('FUNDING_TIMESTAMP_PARSER_SELF_TEST PASS', got.tolist(), flush=True)


def fetch_funding_day_fixed(day):
    url = base.funding_url(day)
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 404:
                return day, None, url, None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.2 * (attempt + 1))
                continue
            r.raise_for_status()
            raw = r.content
            z = zipfile.ZipFile(io.BytesIO(raw))
            names = z.namelist()
            if not names:
                return day, None, url, None
            df = pd.read_csv(io.BytesIO(z.read(names[0])))
            required = {'instrument_name', 'funding_rate', 'funding_time'}
            if not required.issubset(df.columns):
                raise RuntimeError(f'funding schema changed: {list(df.columns)}')

            df = df[['instrument_name', 'funding_rate', 'funding_time']].copy()
            df['funding_time'] = _funding_time_to_epoch_ms(df['funding_time'])
            df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')
            df = df.dropna(subset=['instrument_name', 'funding_rate', 'funding_time'])
            df['funding_time'] = df['funding_time'].astype('int64')

            if len(df):
                lo = int(df['funding_time'].min())
                hi = int(df['funding_time'].max())
                if lo < MIN_VALID_EPOCH_MS or hi > MAX_VALID_EPOCH_MS:
                    raise RuntimeError(f'funding timestamp unit/range invalid: {lo}..{hi}')
            return day, df, url, hashlib.sha256(raw).hexdigest()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(0.8 * (attempt + 1))


_original_load_funding = base.load_funding
base.fetch_funding_day = fetch_funding_day_fixed


def load_funding_verified(symbols):
    allf, provenance, misses = _original_load_funding(symbols)
    rows = int(len(allf))
    n_symbols = int(allf['instrument_name'].nunique()) if rows else 0
    first_ts = int(allf['funding_time'].min()) if rows else 0
    last_ts = int(allf['funding_time'].max()) if rows else 0
    print(
        'FUNDING_INTEGRITY',
        {'rows': rows, 'symbols': n_symbols, 'first_ts': first_ts,
         'last_ts': last_ts, 'archive_days': len(provenance), 'misses': misses},
        flush=True,
    )
    if rows < MIN_FUNDING_ROWS:
        raise RuntimeError(f'funding integrity failed: only {rows} real parsed rows')
    if n_symbols < MIN_FUNDING_SYMBOLS:
        raise RuntimeError(f'funding integrity failed: only {n_symbols} universe symbols')
    if not (MIN_VALID_EPOCH_MS <= first_ts <= MAX_VALID_EPOCH_MS):
        raise RuntimeError(f'funding integrity failed: first timestamp unit/range invalid {first_ts}')
    if not (MIN_VALID_EPOCH_MS <= last_ts <= MAX_VALID_EPOCH_MS):
        raise RuntimeError(f'funding integrity failed: last timestamp unit/range invalid {last_ts}')
    if first_ts > EARLY_COVERAGE_MS:
        raise RuntimeError(f'funding integrity failed: history starts too late at {first_ts}')
    if last_ts < LATE_COVERAGE_MS:
        raise RuntimeError(f'funding integrity failed: history ends too early at {last_ts}')
    return allf, provenance, misses


base.load_funding = load_funding_verified


if __name__ == '__main__':
    _self_test_timestamp_parser()
    base.main()
