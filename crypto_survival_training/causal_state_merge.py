from __future__ import annotations

"""Leakage-resistant joins between completed auxiliary market states and decisions."""

import numpy as np
import pandas as pd


def merge_completed_state(
    decisions: pd.DataFrame,
    state: pd.DataFrame,
    decision_time_col: str = 'decision_time_ms',
    state_time_col: str = 'state_time_ms',
    prefix: str = 'micro_',
    max_age_ms: int | None = None,
) -> pd.DataFrame:
    """Attach only the most recent auxiliary state completed at/before each decision.

    The output contains `<prefix>available`, `<prefix>state_time_ms` and
    `<prefix>age_ms`. A caller may impose a maximum age; stale rows then retain
    the decision but have all auxiliary features cleared and availability=0.
    """
    if decision_time_col not in decisions.columns:
        raise KeyError(decision_time_col)
    if state_time_col not in state.columns:
        raise KeyError(state_time_col)

    left=decisions.copy()
    right=state.copy()
    left[decision_time_col]=pd.to_numeric(left[decision_time_col],errors='raise').astype('int64')
    right[state_time_col]=pd.to_numeric(right[state_time_col],errors='raise').astype('int64')
    left['_merge_order']=np.arange(len(left),dtype=np.int64)
    left=left.sort_values(decision_time_col)
    right=right.sort_values(state_time_col).drop_duplicates(state_time_col,keep='last')

    feature_cols=[c for c in right.columns if c!=state_time_col]
    renamed={state_time_col:prefix+'state_time_ms', **{c:prefix+c for c in feature_cols}}
    right=right.rename(columns=renamed)
    merged=pd.merge_asof(
        left,right,
        left_on=decision_time_col,right_on=prefix+'state_time_ms',
        direction='backward',allow_exact_matches=True,
    )
    matched=merged[prefix+'state_time_ms'].notna()
    merged[prefix+'available']=matched.astype(float)
    merged[prefix+'age_ms']=np.where(
        matched,
        merged[decision_time_col]-merged[prefix+'state_time_ms'],
        np.nan,
    )

    # Hard chronology invariant: an auxiliary state from the future is never legal.
    bad=matched & (merged[prefix+'state_time_ms']>merged[decision_time_col])
    if bad.any():
        raise RuntimeError('causal state merge selected future information')

    if max_age_ms is not None:
        stale=matched & (merged[prefix+'age_ms']>int(max_age_ms))
        if stale.any():
            clear=[prefix+c for c in feature_cols]
            merged.loc[stale,clear]=np.nan
            merged.loc[stale,prefix+'state_time_ms']=np.nan
            merged.loc[stale,prefix+'age_ms']=np.nan
            merged.loc[stale,prefix+'available']=0.0

    return merged.sort_values('_merge_order').drop(columns='_merge_order').reset_index(drop=True)
