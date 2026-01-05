"""
시그널 계산 모듈

전체 시계열 및 특정 시점의 시그널 계산 함수들을 제공합니다.
- Momentum (13612MR, R-squared)
- MACD Histogram
- Correlation
"""

import pandas as pd
import numpy as np
from typing import Tuple
from core.finance import calculate_corr_matrix
from core.models import LM
from core.config import settings


# ============================================================
# 상수 정의
# ============================================================

# 모멘텀 지표 설정
MOMENTUM_PERIODS = (1, 3, 6, 12)  # 13612MR 계산에 사용되는 기간
RSQUARED_PERIODS = (3, 6, 12)  # R² 계산 기간
MONTHS_PER_YEAR = 12  # 연간 개월 수

# MACD 설정
MACD_FAST_PERIOD = 12  # MACD 단기 EMA 기간
MACD_SLOW_PERIOD = 26  # MACD 장기 EMA 기간
MACD_SIGNAL_PERIOD = 9  # MACD Signal Line EMA 기간


# ============================================================
# Signal 계산 (전체 시계열)
# ============================================================

def calculate_all_momentum(closeM: pd.DataFrame, closeM_log: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    전체 시계열의 모멘텀 지표 계산 (lookahead bias 방지)
    pandas 벡터 연산 사용으로 고속 처리

    Parameters:
    -----------
    closeM : pd.DataFrame
        가격 데이터 (rows=dates, cols=tickers)
    closeM_log : pd.DataFrame
        로그 가격 데이터
    verbose : bool
        진행 상황 출력 여부

    Returns:
    --------
    tuple
        (mmt_13612MR, rs_3, rs_6, rs_12) - 각각 (rows=dates, cols=tickers)
    """
    if verbose:
        print(f"      모멘텀 지표 계산 중... (벡터 연산)")

    # 1. 13612MR 계산 (벡터 연산, lookahead bias 자동 방지)
    mr_1 = closeM.pct_change(periods=MOMENTUM_PERIODS[0])
    mr_3 = closeM.pct_change(periods=MOMENTUM_PERIODS[1])
    mr_6 = closeM.pct_change(periods=MOMENTUM_PERIODS[2])
    mr_12 = closeM.pct_change(periods=MOMENTUM_PERIODS[3])

    mmt_13612MR = (mr_1 + mr_3 + mr_6 + mr_12) / 4

    if verbose:
        print(f"        ✓ 13612MR 계산 완료")

    # 2. mean-R² 계산: (1 + √RS3 + √RS6 + √RS12) / 4
    # Rolling window로 각 period별 R² 계산 (최적화: 벡터 연산)
    rs_3 = pd.DataFrame(index=closeM_log.index, columns=closeM_log.columns, dtype=float)
    rs_6 = pd.DataFrame(index=closeM_log.index, columns=closeM_log.columns, dtype=float)
    rs_12 = pd.DataFrame(index=closeM_log.index, columns=closeM_log.columns, dtype=float)

    if verbose:
        print(f"        R² 계산 중... (numpy 벡터 연산)")

    # 각 period별로 rolling R² 계산 (완전 벡터화: ticker + 시점)
    for period, rs_result in zip(RSQUARED_PERIODS, [rs_3, rs_6, rs_12]):
        if len(closeM_log) < period:
            continue

        # X는 모든 window에 공통
        x = np.arange(period)
        x_mean = (period - 1) / 2
        x_centered = x - x_mean
        x_var = np.sum(x_centered ** 2)

        # Sliding window 생성: (n_windows, n_tickers, period)
        from numpy.lib.stride_tricks import sliding_window_view
        data = closeM_log.values  # (n_dates, n_tickers)
        windows = sliding_window_view(data, window_shape=period, axis=0)  # (n_dates-period+1, n_tickers, period)

        # 모든 window와 ticker에 대해 동시에 계산
        # NaN 경고 억제 (NaN만 있는 window는 자동으로 NaN 결과)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            # y: (n_windows, n_tickers, period)
            y_mean = np.nanmean(windows, axis=2, keepdims=True)  # (n_windows, n_tickers, 1)
            y_centered = windows - y_mean  # (n_windows, n_tickers, period)

            # Slope: (n_windows, n_tickers)
            numerator = np.nansum(x_centered * y_centered, axis=2)  # (n_windows, n_tickers)
            slope = numerator / x_var  # (n_windows, n_tickers)

            # R² 계산
            # y_pred: (n_windows, n_tickers, period)
            y_pred = slope[:, :, np.newaxis] * x + (y_mean - slope[:, :, np.newaxis] * x_mean)
            ss_res = np.nansum((windows - y_pred) ** 2, axis=2)  # (n_windows, n_tickers)
            ss_tot = np.nansum(y_centered ** 2, axis=2)  # (n_windows, n_tickers)

            # R² = 1 - (ss_res / ss_tot), ss_tot이 0에 가까우면 0으로 처리
            r2 = 1 - (ss_res / ss_tot)
            r2 = np.where(ss_tot > 1e-10, r2, np.nan)  # 유효하지 않으면 NaN

        # DataFrame에 저장 (날짜 인덱스 맞춤)
        result_dates = closeM_log.index[period - 1:]  # period-1부터 시작
        rs_result.loc[result_dates, :] = r2

    if verbose:
        print(f"        ✓ R² 계산 완료")

    if verbose:
        print(f"      모멘텀 지표 계산 완료!")

    return mmt_13612MR, rs_3, rs_6, rs_12


def calculate_all_macd(closeM: pd.DataFrame, fast_period: int = MACD_FAST_PERIOD, slow_period: int = MACD_SLOW_PERIOD, signal_period: int = MACD_SIGNAL_PERIOD, verbose: bool = True) -> pd.DataFrame:
    """
    전체 시계열의 MACD Histogram 계산 (lookahead bias 방지)
    DataFrame 전체 벡터 연산으로 고속 처리

    Parameters:
    -----------
    closeM : pd.DataFrame
        가격 데이터 (rows=dates, cols=tickers)
    fast_period : int
        단기 EMA 기간
    slow_period : int
        장기 EMA 기간
    signal_period : int
        Signal Line EMA 기간
    verbose : bool
        진행 상황 출력 여부

    Returns:
    --------
    pd.DataFrame
        MACD Histogram (rows=dates, cols=tickers)
    """
    if verbose:
        print(f"      MACD Histogram 계산 중... (전체 DataFrame 벡터 연산)")

    # EMA 계산 (전체 DataFrame에 대해 한번에, ewm은 자동으로 lookahead bias 방지)
    ema_fast = closeM.ewm(span=fast_period, adjust=False).mean()
    ema_slow = closeM.ewm(span=slow_period, adjust=False).mean()

    # MACD Line = EMA(fast) - EMA(slow)
    macd_line = ema_fast - ema_slow

    # Signal Line = EMA(signal_period) of MACD Line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # MACD Histogram = MACD Line - Signal Line
    macd_histogram = macd_line - signal_line

    if verbose:
        print(f"      MACD Histogram 계산 완료!")

    return macd_histogram


# ============================================================
# Signal 계산 (단일 시점 - 하위 호환성 유지)
# ============================================================

def calculate_signals_at_date(closeM_log: pd.DataFrame, closeM: pd.DataFrame, end_idx: int, include_macd: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    특정 시점까지의 데이터로 momentum과 correlation 계산

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
        MACD Histogram 계산 여부 (기본값: True)

    Returns:
    --------
    tuple
        (momentum DataFrame, correlation DataFrame)
    """
    # 해당 시점까지의 데이터만 사용
    prices_log = closeM_log.iloc[:end_idx+1]
    prices = closeM.iloc[:end_idx+1]

    # Momentum DataFrame 초기화
    momentum = pd.DataFrame(index=prices.columns)

    # 1. Core momentum indicators: 벡터 연산으로 계산 후 마지막 값만 추출
    mmt_13612MR_ts, rs_3_ts, rs_6_ts, rs_12_ts = calculate_all_momentum(
        prices, prices_log, verbose=False
    )

    momentum['13612MR'] = mmt_13612MR_ts.iloc[-1]
    momentum['RS3'] = rs_3_ts.iloc[-1]
    momentum['RS6'] = rs_6_ts.iloc[-1]
    momentum['RS12'] = rs_12_ts.iloc[-1]

    # 2. 개별 월별 수익률 (1~12MR): 벡터 연산
    for i in range(1, 13):
        if len(prices) >= i + 1:
            momentum[f'{i}MR'] = prices.pct_change(periods=i).iloc[-1]
        else:
            momentum[f'{i}MR'] = np.nan

    # 3. Annualized Slope (AS): LM 모델 사용 (RSQUARED_PERIODS만)
    for period in RSQUARED_PERIODS:
        if len(prices_log) >= period:
            LR = LM().fit(prices_log, period)
            momentum[f'AS{period}'] = (np.exp(LR.slope * MONTHS_PER_YEAR) - 1)
        else:
            momentum[f'AS{period}'] = np.nan

    # 4. MACD Histogram: 벡터 연산으로 계산 후 마지막 값만 추출
    if include_macd:
        if len(prices) >= MACD_SLOW_PERIOD:
            macd_hist_ts = calculate_all_macd(
                prices,
                fast_period=MACD_FAST_PERIOD,
                slow_period=MACD_SLOW_PERIOD,
                signal_period=MACD_SIGNAL_PERIOD,
                verbose=False
            )
            momentum['MACD_Histogram'] = macd_hist_ts.iloc[-1]
        else:
            momentum['MACD_Histogram'] = np.nan
    else:
        momentum['MACD_Histogram'] = np.nan

    # 5. Correlation 계산
    corr_periods = settings.signals.correlation.periods
    if len(prices) >= corr_periods:
        correlation = calculate_corr_matrix(prices, corr_periods)
    else:
        # 데이터 부족 시 빈 상관관계 행렬
        correlation = pd.DataFrame(index=prices.columns, columns=prices.columns)
        correlation[:] = np.nan

    return momentum, correlation
