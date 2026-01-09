"""
시그널 계산 모듈

전체 시계열 및 특정 시점의 시그널 계산 함수들을 제공합니다.
- Momentum (13612MR, R-squared)
- MACD Histogram
- Correlation
"""

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import warnings
from typing import Tuple
from core.finance import calculate_corr_matrix
from core.models import LM
from core.config import settings


# ============================================================
# 내부 상수 (Minor Constants)
# ============================================================

_MONTHS_PER_YEAR = 12


# ============================================================
# Layer 0: 내부 계산 함수 (Private)
# ============================================================

def _calculate_13612mr(closeM: pd.DataFrame) -> pd.DataFrame:
    """
    13612MR 계산 (순수 함수).

    Parameters:
    -----------
    closeM : pd.DataFrame
        가격 데이터 (rows=dates, cols=tickers)

    Returns:
    --------
    pd.DataFrame
        13612MR 시계열 데이터
    """
    periods = tuple(settings.signals.momentum.calculation_periods)

    mr_values = [closeM.pct_change(periods=period) for period in periods]
    mmt_13612MR = sum(mr_values) / len(mr_values)

    return mmt_13612MR


def _calculate_rsquared_rolling(
    closeM_log: pd.DataFrame,
    periods: tuple
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Rolling R² 계산 (순수 함수, 벡터화).

    Parameters:
    -----------
    closeM_log : pd.DataFrame
        로그 가격 데이터
    periods : tuple
        R² 계산 기간 (예: (3, 6, 12))

    Returns:
    --------
    tuple
        (rs_3, rs_6, rs_12) DataFrame들
    """
    results = []

    for period in periods:
        rs_result = pd.DataFrame(
            index=closeM_log.index,
            columns=closeM_log.columns,
            dtype=float
        )

        if len(closeM_log) < period:
            results.append(rs_result)
            continue

        # X는 모든 window에 공통
        x = np.arange(period)
        x_mean = (period - 1) / 2
        x_centered = x - x_mean
        x_var = np.sum(x_centered ** 2)

        # Sliding window 생성
        data = closeM_log.values
        windows = sliding_window_view(data, window_shape=period, axis=0)

        # 벡터 연산 (모든 window와 ticker 동시 처리)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            y_mean = np.nanmean(windows, axis=2, keepdims=True)
            y_centered = windows - y_mean

            # Slope 계산
            numerator = np.nansum(x_centered * y_centered, axis=2)
            slope = numerator / x_var

            # R² 계산
            y_pred = slope[:, :, np.newaxis] * x + (y_mean - slope[:, :, np.newaxis] * x_mean)
            ss_res = np.nansum((windows - y_pred) ** 2, axis=2)
            ss_tot = np.nansum(y_centered ** 2, axis=2)

            r2 = 1 - (ss_res / ss_tot)
            r2 = np.where(ss_tot > 1e-10, r2, np.nan)

        # DataFrame에 저장
        result_dates = closeM_log.index[period - 1:]
        rs_result.loc[result_dates, :] = r2

        results.append(rs_result)

    return tuple(results)


def _calculate_macd_histogram(
    closeM: pd.DataFrame,
    fast_period: int,
    slow_period: int,
    signal_period: int
) -> pd.DataFrame:
    """
    MACD Histogram 계산 (순수 함수).

    Parameters:
    -----------
    closeM : pd.DataFrame
        가격 데이터
    fast_period : int
        단기 EMA 기간
    slow_period : int
        장기 EMA 기간
    signal_period : int
        Signal Line EMA 기간

    Returns:
    --------
    pd.DataFrame
        MACD Histogram 시계열
    """
    ema_fast = closeM.ewm(span=fast_period, adjust=False).mean()
    ema_slow = closeM.ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line

    return macd_histogram


def _build_momentum_dataframe(
    prices: pd.DataFrame,
    prices_log: pd.DataFrame,
    include_macd: bool
) -> pd.DataFrame:
    """
    모멘텀 DataFrame 구성 (내부 함수).

    Parameters:
    -----------
    prices : pd.DataFrame
        가격 데이터
    prices_log : pd.DataFrame
        로그 가격 데이터
    include_macd : bool
        MACD Histogram 포함 여부

    Returns:
    --------
    pd.DataFrame
        모멘텀 지표 DataFrame
    """
    momentum = pd.DataFrame(index=prices.columns)

    # 1. Core momentum indicators
    mmt_13612MR_ts, rs_3_ts, rs_6_ts, rs_12_ts = calculate_all_momentum(prices, prices_log)

    momentum['13612MR'] = mmt_13612MR_ts.iloc[-1]
    momentum['RS3'] = rs_3_ts.iloc[-1]
    momentum['RS6'] = rs_6_ts.iloc[-1]
    momentum['RS12'] = rs_12_ts.iloc[-1]

    # 2. 개별 월별 수익률 (1~12MR)
    for i in range(1, 13):
        if len(prices) >= i + 1:
            momentum[f'{i}MR'] = prices.pct_change(periods=i).iloc[-1]
        else:
            momentum[f'{i}MR'] = np.nan

    # 3. Annualized Slope (AS)
    rsquared_periods = tuple(settings.signals.momentum.rsquared_periods)
    for period in rsquared_periods:
        if len(prices_log) >= period:
            LR = LM().fit(prices_log, period)
            momentum[f'AS{period}'] = (np.exp(LR.slope * _MONTHS_PER_YEAR) - 1)
        else:
            momentum[f'AS{period}'] = np.nan

    # 4. MACD Histogram
    if include_macd:
        macd_slow = settings.signals.macd.slow_period
        if len(prices) >= macd_slow:
            macd_hist_ts = calculate_all_macd(prices)
            momentum['MACD_Histogram'] = macd_hist_ts.iloc[-1]
        else:
            momentum['MACD_Histogram'] = np.nan
    else:
        momentum['MACD_Histogram'] = np.nan

    return momentum


def _build_correlation_dataframe(prices: pd.DataFrame) -> pd.DataFrame:
    """
    상관관계 DataFrame 구성 (내부 함수).

    Parameters:
    -----------
    prices : pd.DataFrame
        가격 데이터

    Returns:
    --------
    pd.DataFrame
        상관관계 행렬
    """
    corr_periods = settings.signals.correlation.periods

    if len(prices) >= corr_periods:
        return calculate_corr_matrix(prices, corr_periods)
    else:
        # 데이터 부족 시 빈 행렬
        correlation = pd.DataFrame(
            index=prices.columns,
            columns=prices.columns
        )
        correlation[:] = np.nan
        return correlation


# ============================================================
# Layer 1: 공개 API (Public Interface)
# ============================================================

def calculate_all_momentum(
    closeM: pd.DataFrame,
    closeM_log: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    전체 시계열의 모멘텀 지표 계산 (lookahead bias 방지).
    pandas 벡터 연산 사용으로 고속 처리.

    Parameters:
    -----------
    closeM : pd.DataFrame
        가격 데이터 (rows=dates, cols=tickers)
    closeM_log : pd.DataFrame
        로그 가격 데이터

    Returns:
    --------
    tuple
        (mmt_13612MR, rs_3, rs_6, rs_12) - 각각 (rows=dates, cols=tickers)
    """
    # 13612MR 계산
    mmt_13612MR = _calculate_13612mr(closeM)

    # R² 계산
    rsquared_periods = tuple(settings.signals.momentum.rsquared_periods)
    rs_results = _calculate_rsquared_rolling(closeM_log, rsquared_periods)

    return (mmt_13612MR,) + rs_results


def calculate_all_macd(
    closeM: pd.DataFrame,
    fast_period: int = None,
    slow_period: int = None,
    signal_period: int = None
) -> pd.DataFrame:
    """
    전체 시계열의 MACD Histogram 계산 (lookahead bias 방지).
    DataFrame 전체 벡터 연산으로 고속 처리.

    Parameters:
    -----------
    closeM : pd.DataFrame
        가격 데이터 (rows=dates, cols=tickers)
    fast_period : int, optional
        단기 EMA 기간 (None이면 설정값 사용)
    slow_period : int, optional
        장기 EMA 기간 (None이면 설정값 사용)
    signal_period : int, optional
        Signal Line EMA 기간 (None이면 설정값 사용)

    Returns:
    --------
    pd.DataFrame
        MACD Histogram (rows=dates, cols=tickers)
    """
    if fast_period is None:
        fast_period = settings.signals.macd.fast_period
    if slow_period is None:
        slow_period = settings.signals.macd.slow_period
    if signal_period is None:
        signal_period = settings.signals.macd.signal_period

    return _calculate_macd_histogram(closeM, fast_period, slow_period, signal_period)


def calculate_signals_at_date(
    closeM_log: pd.DataFrame,
    closeM: pd.DataFrame,
    end_idx: int,
    include_macd: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    특정 시점까지의 데이터로 momentum과 correlation 계산.

    내부적으로 벡터 연산(calculate_all_momentum, calculate_all_macd)을 활용하여
    코드 중복을 제거하고 성능을 개선합니다.

    Parameters:
    -----------
    closeM_log : pd.DataFrame
        로그 가격 데이터
    closeM : pd.DataFrame
        가격 데이터
    end_idx : int
        계산할 마지막 인덱스
    include_macd : bool
        MACD Histogram 계산 여부

    Returns:
    --------
    tuple
        (momentum DataFrame, correlation DataFrame)
    """
    # 해당 시점까지의 데이터만 사용
    # 음수 인덱스 처리: -1은 마지막 행을 의미
    if end_idx < 0:
        actual_idx = len(closeM) + end_idx
    else:
        actual_idx = end_idx

    prices_log = closeM_log.iloc[:actual_idx + 1]
    prices = closeM.iloc[:actual_idx + 1]

    # 모멘텀과 상관관계 계산
    momentum = _build_momentum_dataframe(prices, prices_log, include_macd)
    correlation = _build_correlation_dataframe(prices)

    return momentum, correlation
