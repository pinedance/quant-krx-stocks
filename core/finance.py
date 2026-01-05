"""금융 지표 계산 함수들"""
import numpy as np
import pandas as pd


def rt(prices, periods):
    """
    Return (수익률) 계산

    Parameters:
    -----------
    prices : pd.DataFrame or pd.Series
        가격 데이터
    periods : int
        기간 (개월 수)

    Returns:
    --------
    pd.Series or float
        수익률
    """
    if isinstance(prices, pd.DataFrame):
        return prices.pct_change(periods=periods).iloc[-1]
    elif isinstance(prices, pd.Series):
        if len(prices) < periods + 1:
            return np.nan
        return (prices.iloc[-1] / prices.iloc[-periods-1]) - 1
    else:
        return (prices[-1] / prices[-periods-1]) - 1


def annualize_rt(rt, periods, period_type='M'):
    """
    값을 연율화합니다.

    Parameters:
    -----------
    rt : float or pd.Series
        연율화할 값(return)
    periods : int
        기간
    period_type : str
        기간 타입 ('M': 월, 'D': 일, 'Y': 년)

    Returns:
    --------
    float or pd.Series
        연율화된 값
    """
    if period_type == 'M':
        factor = 12 / periods
    elif period_type == 'D':
        factor = 252 / periods
    elif period_type == 'Y':
        factor = 1 / periods
    else:
        raise ValueError(f"Unknown period_type: {period_type}")

    return ( (rt + 1) ** factor ) - 1


def stdev(prices, periods):
    """
    Standard Deviation (표준편차) 계산 및 연율화

    Parameters:
    -----------
    prices : pd.DataFrame or pd.Series
        가격 데이터
    periods : int
        기간 (개월 수)

    Returns:
    --------
    float or pd.Series
        연율화된 표준편차
    """
    if isinstance(prices, pd.DataFrame):
        returns = prices.pct_change().iloc[-periods:]
        monthly_std = returns.std(ddof=1)
        return monthly_std * np.sqrt(12)  # 연율화
    elif isinstance(prices, pd.Series):
        if len(prices) < periods + 1:
            return np.nan
        returns = prices.pct_change().iloc[-periods:]
        monthly_std = returns.std(ddof=1)
        return monthly_std * np.sqrt(12)  # 연율화
    else:
        raise ValueError("prices must be DataFrame or Series")


def dsdev(prices, periods):
    """
    Downside Deviation (하방 편차) 계산 및 연율화

    Parameters:
    -----------
    prices : pd.DataFrame or pd.Series
        가격 데이터
    periods : int
        기간 (개월 수)

    Returns:
    --------
    float or pd.Series
        연율화된 하방 편차
    """
    if isinstance(prices, pd.DataFrame):
        returns = prices.pct_change().iloc[-periods:]
        # 각 컬럼별로 downside deviation 계산
        result = pd.Series(index=returns.columns, dtype=float)
        for col in returns.columns:
            col_returns = returns[col].dropna()
            downside = col_returns[col_returns < 0]
            if len(downside) > 1:
                result[col] = downside.std(ddof=1) * np.sqrt(12)  # 연율화
            else:
                result[col] = np.nan
        return result

    elif isinstance(prices, pd.Series):
        if len(prices) < periods + 1:
            return np.nan
        returns = prices.pct_change().iloc[-periods:].dropna()
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1:
            return downside_returns.std(ddof=1) * np.sqrt(12)  # 연율화
        else:
            return np.nan
    else:
        raise ValueError("prices must be DataFrame or Series")


def calculate_corr_matrix(prices, periods):
    """
    상관계수 행렬 계산

    Parameters:
    -----------
    prices : pd.DataFrame
        가격 데이터 (columns: tickers, index: dates)
    periods : int
        기간 (개월 수)

    Returns:
    --------
    pd.DataFrame
        상관계수 행렬 + marginal_sum
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be DataFrame")

    ncol = len(prices.columns)

    # 최근 periods 기간의 수익률 계산
    returns = prices.pct_change(fill_method=None).iloc[-periods:]

    # 상관계수 행렬 계산
    corr_matrix = returns.corr()
    # marginal mean 추가 (각 종목의 다른 종목들과의 상관계수 평균, 자기 자신 제외)
    corr_matrix['mean'] = (corr_matrix.sum(axis=1) - 1) / (ncol-1)
    corr_matrix.loc['mean'] = (corr_matrix.sum(axis=0) - 1) / (ncol-1)

    return corr_matrix


# ============================================================
# 파생 지표 계산 (Derived Metrics)
# ============================================================

def calculate_momentum_quality(momentum, period):
    """
    Momentum Quality (√R² × AS) 계산

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


def calculate_sharpe_ratio(performance, period):
    """
    Sharpe Ratio (AR / SD) 계산

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


def calculate_sortino_ratio(performance, period):
    """
    Sortino Ratio (AR / DD) 계산

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


def calculate_correlation_coefficient(momentum, period):
    """
    Correlation Coefficient (√R²) 계산

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
