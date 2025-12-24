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


def annualize(value, periods, period_type='M'):
    """
    값을 연율화합니다.

    Parameters:
    -----------
    value : float or pd.Series
        연율화할 값
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

    return value * factor


def stdev(prices, periods):
    """
    Standard Deviation (표준편차) 계산

    Parameters:
    -----------
    prices : pd.DataFrame or pd.Series
        가격 데이터
    periods : int
        기간 (개월 수)

    Returns:
    --------
    float or pd.Series
        표준편차
    """
    if isinstance(prices, pd.DataFrame):
        returns = prices.pct_change().iloc[-periods:]
        return returns.std(ddof=1)
    elif isinstance(prices, pd.Series):
        if len(prices) < periods + 1:
            return np.nan
        returns = prices.pct_change().iloc[-periods:]
        return returns.std(ddof=1)
    else:
        raise ValueError("prices must be DataFrame or Series")


def dsdev(prices, periods):
    """
    Downside Deviation (하방 편차) 계산

    Parameters:
    -----------
    prices : pd.DataFrame or pd.Series
        가격 데이터
    periods : int
        기간 (개월 수)

    Returns:
    --------
    float or pd.Series
        하방 편차
    """
    if isinstance(prices, pd.DataFrame):
        returns = prices.pct_change().iloc[-periods:]
        # 각 컬럼별로 downside deviation 계산
        result = pd.Series(index=returns.columns, dtype=float)
        for col in returns.columns:
            col_returns = returns[col].dropna()
            downside = col_returns[col_returns < 0]
            result[col] = downside.std(ddof=1) if len(downside) > 1 else np.nan
        return result

    elif isinstance(prices, pd.Series):
        if len(prices) < periods + 1:
            return np.nan
        returns = prices.pct_change().iloc[-periods:].dropna()
        downside_returns = returns[returns < 0]
        return downside_returns.std(ddof=1) if len(downside_returns) > 1 else np.nan
    else:
        raise ValueError("prices must be DataFrame or Series")


def get_corrMatrix(prices, periods):
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

    # 최근 periods 기간의 수익률 계산
    returns = prices.pct_change().iloc[-periods:]

    # 상관계수 행렬 계산
    corr_matrix = returns.corr()

    # marginal sum 추가 (각 종목의 다른 종목들과의 상관계수 합)
    corr_matrix['marginal_sum'] = corr_matrix.sum(axis=1) - 1  # 자기 자신과의 상관계수 1 제외
    corr_matrix.loc['marginal_sum'] = corr_matrix.sum(axis=0) - 1

    return corr_matrix
