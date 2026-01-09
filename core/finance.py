"""
금융 지표 계산 모듈

기본 금융 지표(수익률, 변동성, 상관계수)와 파생 지표(샤프/소르티노 비율, 모멘텀 퀄리티)를 제공합니다.
- DataFrame 전용 API (타입 분기 제거)
- 레이어 아키텍처 (Layer 0: 순수 계산, Layer 1: 공개 API)
- 수학적 정확성 (기하평균 연율화, 분산 스케일링)
"""

import numpy as np
import pandas as pd


# ============================================================
# 내부 상수 (Minor Constants)
# ============================================================

_MONTHS_PER_YEAR = 12
_TRADING_DAYS_PER_YEAR = 252
_DDOF = 1  # 표본 표준편차 (Degrees of Freedom)


# ============================================================
# Layer 0: 내부 계산 함수 (Private)
# ============================================================

def _annualize_return(rt: float, factor: float) -> float:
    """
    수익률을 연율화합니다 (기하평균 방식).

    수학적 배경:
    - 복리 효과를 고려한 정확한 연율화
    - (1 + r)^factor - 1

    Parameters:
    -----------
    rt : float
        수익률
    factor : float
        연율화 팩터 (예: 12개월 수익률 → factor=1, 6개월 → factor=2)

    Returns:
    --------
    float
        연율화된 수익률
    """
    return ((rt + 1) ** factor) - 1


def _annualize_volatility(monthly_vol: float) -> float:
    """
    월간 변동성을 연율화합니다 (분산 스케일링 방식).

    수학적 배경:
    - 분산은 시간에 비례하여 스케일링
    - 표준편차는 √시간에 비례: σ_annual = σ_monthly × √12

    Parameters:
    -----------
    monthly_vol : float
        월간 변동성

    Returns:
    --------
    float
        연율화된 변동성
    """
    return monthly_vol * np.sqrt(_MONTHS_PER_YEAR)


def _get_annualization_factor(periods: int, period_type: str) -> float:
    """
    기간 타입에 따른 연율화 팩터를 반환합니다.

    Parameters:
    -----------
    periods : int
        기간
    period_type : str
        기간 타입 ('M': 월, 'D': 일, 'Y': 년)

    Returns:
    --------
    float
        연율화 팩터
    """
    if period_type == 'M':
        return _MONTHS_PER_YEAR / periods
    elif period_type == 'D':
        return _TRADING_DAYS_PER_YEAR / periods
    elif period_type == 'Y':
        return 1 / periods
    else:
        raise ValueError(f"Unknown period_type: {period_type}")


def _calculate_std_from_returns(returns: pd.DataFrame) -> pd.Series:
    """
    수익률로부터 표준편차를 계산합니다 (순수 함수).

    Parameters:
    -----------
    returns : pd.DataFrame
        수익률 데이터

    Returns:
    --------
    pd.Series
        각 종목의 표준편차
    """
    return returns.std(ddof=_DDOF)


def _calculate_downside_std(returns: pd.DataFrame) -> pd.Series:
    """
    수익률로부터 하방 표준편차를 계산합니다 (순수 함수).

    하방 편차는 음수 수익률만을 사용하여 계산합니다.

    Parameters:
    -----------
    returns : pd.DataFrame
        수익률 데이터

    Returns:
    --------
    pd.Series
        각 종목의 하방 표준편차
    """
    result = pd.Series(index=returns.columns, dtype=float)

    for col in returns.columns:
        col_returns = returns[col].dropna()
        downside = col_returns[col_returns < 0]

        if len(downside) > 1:
            result[col] = downside.std(ddof=_DDOF)
        else:
            result[col] = np.nan

    return result


# ============================================================
# Layer 1: 공개 API (Public Interface)
# ============================================================

def rt(prices: pd.DataFrame, periods: int) -> pd.Series:
    """
    Return (수익률) 계산.

    가장 최근 시점의 수익률을 반환합니다.

    Parameters:
    -----------
    prices : pd.DataFrame
        가격 데이터 (rows=dates, cols=tickers)
    periods : int
        기간 (개월 수)

    Returns:
    --------
    pd.Series
        각 종목의 수익률 (가장 최근 시점)
    """
    return prices.pct_change(periods=periods).iloc[-1]


def annualize_rt(rt, periods: int, period_type: str = 'M'):
    """
    수익률을 연율화합니다.

    복리 효과를 고려한 기하평균 방식을 사용합니다.
    예: 6개월 수익률 10% → 연율화: (1.1)^2 - 1 = 21%

    Parameters:
    -----------
    rt : float or pd.Series
        연율화할 수익률
    periods : int
        기간
    period_type : str
        기간 타입 ('M': 월, 'D': 일, 'Y': 년)

    Returns:
    --------
    float or pd.Series
        연율화된 수익률
    """
    factor = _get_annualization_factor(periods, period_type)

    if isinstance(rt, pd.Series):
        return rt.apply(lambda r: _annualize_return(r, factor))
    else:
        return _annualize_return(rt, factor)


def stdev(prices: pd.DataFrame, periods: int) -> pd.Series:
    """
    Standard Deviation (표준편차) 계산 및 연율화.

    최근 periods 개월의 월간 수익률로부터 표준편차를 계산하고,
    √12를 곱하여 연율화합니다.

    Parameters:
    -----------
    prices : pd.DataFrame
        가격 데이터 (rows=dates, cols=tickers)
    periods : int
        기간 (개월 수)

    Returns:
    --------
    pd.Series
        각 종목의 연율화된 표준편차
    """
    returns = prices.pct_change().iloc[-periods:]
    monthly_std = _calculate_std_from_returns(returns)
    return monthly_std.apply(_annualize_volatility)


def dsdev(prices: pd.DataFrame, periods: int) -> pd.Series:
    """
    Downside Deviation (하방 편차) 계산 및 연율화.

    최근 periods 개월의 월간 수익률 중 음수 수익률만 사용하여
    하방 표준편차를 계산하고 연율화합니다.

    하방 편차는 하방 리스크(손실 변동성)를 측정하는 지표로,
    소르티노 비율(Sortino Ratio) 계산에 사용됩니다.

    Parameters:
    -----------
    prices : pd.DataFrame
        가격 데이터 (rows=dates, cols=tickers)
    periods : int
        기간 (개월 수)

    Returns:
    --------
    pd.Series
        각 종목의 연율화된 하방 편차
    """
    returns = prices.pct_change().iloc[-periods:]
    downside_std = _calculate_downside_std(returns)
    return downside_std.apply(_annualize_volatility)


def calculate_corr_matrix(prices: pd.DataFrame, periods: int) -> pd.DataFrame:
    """
    상관계수 행렬 계산.

    최근 periods 개월의 수익률로부터 종목 간 상관계수 행렬을 계산하고,
    각 종목의 평균 상관계수(marginal mean)를 추가합니다.

    Parameters:
    -----------
    prices : pd.DataFrame
        가격 데이터 (rows=dates, cols=tickers)
    periods : int
        기간 (개월 수)

    Returns:
    --------
    pd.DataFrame
        상관계수 행렬 + mean 행/열
        - mean 열: 각 종목의 다른 종목들과의 평균 상관계수
        - mean 행: 각 종목의 다른 종목들과의 평균 상관계수 (열과 동일)
    """
    ncol = len(prices.columns)

    # 최근 periods 기간의 수익률 계산
    returns = prices.pct_change(fill_method=None).iloc[-periods:]

    # 상관계수 행렬 계산
    corr_matrix = returns.corr()

    # marginal mean 추가
    # (각 종목의 다른 종목들과의 상관계수 평균, 자기 자신 제외)
    # sum - 1: 대각선 요소(자기 자신과의 상관계수 1.0) 제외
    # / (ncol - 1): 나머지 종목 수로 나누어 평균 계산
    corr_matrix['mean'] = (corr_matrix.sum(axis=1) - 1) / (ncol - 1)
    corr_matrix.loc['mean'] = (corr_matrix.sum(axis=0) - 1) / (ncol - 1)

    return corr_matrix


# ============================================================
# 파생 지표 계산 (Derived Metrics)
# ============================================================

def calculate_momentum_quality(momentum: pd.DataFrame, period: int) -> pd.Series:
    """
    Momentum Quality (√R² × AS) 계산.

    모멘텀의 신뢰도(R²)와 크기(AS)를 결합한 지표입니다.
    - R²이 높을수록: 추세가 명확
    - AS가 클수록: 상승 모멘텀이 강함

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 데이터 (RS{period}, AS{period} 컬럼 필요)
    period : int
        기간 (개월)

    Returns:
    --------
    pd.Series
        Momentum Quality 값
    """
    return np.sqrt(momentum[f'RS{period}']) * momentum[f'AS{period}']


def calculate_sharpe_ratio(performance: pd.DataFrame, period: int) -> pd.Series:
    """
    Sharpe Ratio (AR / SD) 계산.

    단위 위험(표준편차)당 수익률을 측정하는 지표입니다.
    값이 클수록 위험 대비 수익이 높습니다.

    Parameters:
    -----------
    performance : pd.DataFrame
        Performance 데이터 (AR{period}, SD{period} 컬럼 필요)
    period : int
        기간 (개월)

    Returns:
    --------
    pd.Series
        Sharpe Ratio 값
    """
    return performance[f'AR{period}'] / performance[f'SD{period}']


def calculate_sortino_ratio(performance: pd.DataFrame, period: int) -> pd.Series:
    """
    Sortino Ratio (AR / DD) 계산.

    단위 하방 위험(하방 편차)당 수익률을 측정하는 지표입니다.
    Sharpe Ratio와 달리 하방 변동성만 고려하여 더 정확한 위험 측정이 가능합니다.

    Parameters:
    -----------
    performance : pd.DataFrame
        Performance 데이터 (AR{period}, DD{period} 컬럼 필요)
    period : int
        기간 (개월)

    Returns:
    --------
    pd.Series
        Sortino Ratio 값
    """
    return performance[f'AR{period}'] / performance[f'DD{period}']


def calculate_correlation_coefficient(momentum: pd.DataFrame, period: int) -> pd.Series:
    """
    Correlation Coefficient (√R²) 계산.

    결정계수(R²)의 제곱근으로, 선형 회귀의 적합도를 나타냅니다.
    값이 1에 가까울수록 추세가 명확합니다.

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 데이터 (RS{period} 컬럼 필요)
    period : int
        기간 (개월)

    Returns:
    --------
    pd.Series
        Correlation Coefficient 값
    """
    return np.sqrt(momentum[f'RS{period}'])
