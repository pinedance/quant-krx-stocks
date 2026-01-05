"""
전략 설정 및 포트폴리오 구성 모듈

전략 설정, 필터링, 포트폴리오 구성 및 비교 함수들을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, List
from dataclasses import dataclass


# ============================================================
# Strategy Configuration
# ============================================================

@dataclass
class StrategyConfig:
    """
    백테스트 전략 설정

    Attributes:
    -----------
    name : str
        전략 이름 (예: 'strategy1')
    momentum_ratio : float
        모멘텀 상위 비율 (0.33 = 상위 1/3, 0.5 = 상위 1/2)
    rsquared_ratio : float
        R-squared 상위 비율
    correlation_ratio : float
        Correlation 하위 비율 (낮을수록 분산 효과 강화)
    use_inverse : bool
        음수 모멘텀 종목에 인버스 ETF 사용 여부
    use_macd_filter : bool
        MACD 오실레이터 필터 사용 여부 (3MR - 12MR)
    description : str
        전략 설명
    """
    name: str
    momentum_ratio: float
    rsquared_ratio: float
    correlation_ratio: float
    use_inverse: bool = False
    use_macd_filter: bool = False
    description: str = ""


# ============================================================
# 지표 계산
# ============================================================

def calculate_avg_momentum(momentum: pd.DataFrame) -> pd.Series:
    """
    13612MR 계산

    Parameters:
    -----------
    momentum : pd.DataFrame
        모멘텀 데이터프레임

    Returns:
    --------
    pd.Series
        13612MR (1MR + 3MR + 6MR + 12MR) / 4
    """
    return momentum['13612MR']


def calculate_avg_rsquared(momentum: pd.DataFrame) -> pd.Series:
    """
    mean-R² 계산: (1 + √RS3 + √RS6 + √RS12) / 4

    Parameters:
    -----------
    momentum : pd.DataFrame
        모멘텀 데이터프레임 (RS3, RS6, RS12 컬럼 필요)

    Returns:
    --------
    pd.Series
        평균 R-squared 값
    """
    return (1 + np.sqrt(momentum['RS3']) +
            np.sqrt(momentum['RS6']) +
            np.sqrt(momentum['RS12'])) / 4


def calculate_marginal_means(correlation: pd.DataFrame, tickers: List[str]) -> pd.Series:
    """
    Correlation marginal mean 계산

    Parameters:
    -----------
    correlation : pd.DataFrame
        상관계수 행렬
    tickers : List[str]
        대상 종목 리스트

    Returns:
    --------
    pd.Series
        각 종목의 marginal mean (다른 종목들과의 평균 상관계수)
    """
    # 'mean' 행/열 제거
    corr_matrix = correlation.drop('mean', axis=0, errors='ignore').drop('mean', axis=1, errors='ignore')

    # 해당 종목들만 필터링
    available_tickers = [t for t in tickers if t in corr_matrix.index]

    if len(available_tickers) == 0:
        return pd.Series(dtype=float)

    sub_corr = corr_matrix.loc[available_tickers, available_tickers]
    n = len(available_tickers)

    if n > 1:
        # 대각선 제외한 평균
        marginal_means = (sub_corr.sum(axis=1) - 1) / (n - 1)
        return marginal_means
    else:
        # 종목이 1개면 marginal mean 없음
        return pd.Series([0.0], index=available_tickers)


# ============================================================
# 필터링
# ============================================================

def apply_filters(
    momentum: pd.DataFrame,
    correlation: pd.DataFrame,
    momentum_ratio: float,
    rsquared_ratio: float,
    correlation_ratio: float
) -> List[str]:
    """
    3단계 필터링 수행

    Parameters:
    -----------
    momentum : pd.DataFrame
        모멘텀 데이터
    correlation : pd.DataFrame
        상관계수 행렬
    momentum_ratio : float
        모멘텀 상위 비율 (예: 0.5 = 상위 50%)
    rsquared_ratio : float
        R-squared 상위 비율
    correlation_ratio : float
        Correlation 하위 비율

    Returns:
    --------
    List[str]
        필터링된 종목 리스트
    """
    total_stocks = len(momentum)

    # Step 1: 평균 모멘텀 필터링
    avg_momentum = calculate_avg_momentum(momentum)
    step1_count = int(total_stocks * momentum_ratio)
    step1_tickers = avg_momentum.nlargest(step1_count).index.tolist()

    # Step 2: 평균 R-squared 필터링
    avg_rsquared = calculate_avg_rsquared(momentum)
    step2_candidates = avg_rsquared[step1_tickers]
    step2_count = int(len(step1_tickers) * rsquared_ratio)
    step2_tickers = step2_candidates.nlargest(step2_count).index.tolist()

    # Step 3: Marginal mean 필터링
    marginal_means = calculate_marginal_means(correlation, step2_tickers)
    if len(marginal_means) == 0:
        return []

    step3_count = int(len(marginal_means) * correlation_ratio)
    step3_tickers = marginal_means.nsmallest(step3_count).index.tolist()

    return step3_tickers


# ============================================================
# 포트폴리오 구성
# ============================================================

def build_portfolio(
    tickers: List[str],
    momentum: pd.DataFrame,
    use_inverse: bool = False,
    use_macd_filter: bool = False
) -> Dict[str, float]:
    """
    포트폴리오 구성

    Parameters:
    -----------
    tickers : List[str]
        선택된 종목 리스트
    momentum : pd.DataFrame
        모멘텀 데이터
    use_inverse : bool
        음수 모멘텀 종목에 인버스 ETF 사용 여부
    use_macd_filter : bool
        MACD 필터 사용 여부

    Returns:
    --------
    Dict[str, float]
        포트폴리오 (ticker: weight)
    """
    if len(tickers) == 0:
        return {}

    n_stocks = len(tickers)
    equal_weight = 1.0 / n_stocks if n_stocks > 0 else 0

    # 벡터 연산으로 필터링
    mom_values = momentum.loc[tickers, '13612MR']

    if use_macd_filter:
        # MACD 필터 적용 (벡터 연산)
        macd_values = momentum.loc[tickers, 'MACD_Histogram']
        # 13612MR >= 0 AND MACD_Histogram >= 0
        valid_mask = (mom_values >= 0) & (macd_values >= 0)
        valid_tickers = mom_values[valid_mask].index.tolist()
        portfolio = {ticker: equal_weight for ticker in valid_tickers}
        inverse_weight = 0.0

    else:
        # 기본 필터만 적용
        positive_mask = mom_values >= 0
        negative_mask = mom_values < 0

        # 양수 모멘텀: 투자
        valid_tickers = mom_values[positive_mask].index.tolist()
        portfolio = {ticker: equal_weight for ticker in valid_tickers}

        # 음수 모멘텀: 인버스 또는 현금
        inverse_weight = 0.0
        if use_inverse:
            n_negative = negative_mask.sum()
            inverse_weight = (equal_weight / 4) * n_negative

    # 인버스 가중치 추가
    if inverse_weight > 0 and use_inverse:
        portfolio['INVERSE'] = inverse_weight

    return portfolio


# ============================================================
# DataFrame 구성
# ============================================================

def build_selected_dataframe(
    tickers: list,
    tickers_info: pd.DataFrame,
    avg_momentum: pd.Series,
    avg_rsquared: pd.Series,
    marginal_means: pd.Series
) -> pd.DataFrame:
    """
    선택된 종목 정보를 DataFrame으로 구성

    Parameters:
    -----------
    tickers : list
        선택된 종목 리스트
    tickers_info : pd.DataFrame
        전체 종목 정보 (Code, Name 컬럼 포함)
    avg_momentum : pd.Series
        평균 모멘텀 (13612MR)
    avg_rsquared : pd.Series
        평균 R-squared
    marginal_means : pd.Series
        Marginal means

    Returns:
    --------
    pd.DataFrame
        선택된 종목 정보 (index: Ticker with 'S' prefix, columns: Name, avg_momentum, avg_rsquared, marginal_mean)
    """
    ticker_to_name = dict(zip(tickers_info['Code'], tickers_info['Name']))

    selected_data = []
    for ticker in tickers:
        selected_data.append({
            'Ticker': f'S{ticker}',
            'Name': ticker_to_name.get(ticker, ''),
            'avg_momentum': avg_momentum[ticker],
            'avg_rsquared': avg_rsquared[ticker],
            'marginal_mean': marginal_means[ticker]
        })

    selected = pd.DataFrame(selected_data)
    selected = selected.set_index('Ticker')

    return selected


def format_portfolio_as_dataframe(
    portfolio_dict: Dict[str, float],
    tickers_info: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    백테스트용 Dict를 출력용 DataFrame으로 변환

    Parameters:
    -----------
    portfolio_dict : Dict[str, float]
        포트폴리오 (ticker: weight) from build_portfolio()
    tickers_info : pd.DataFrame
        종목 정보 (Code, Name 컬럼 포함)
    verbose : bool
        요약 정보 출력 여부

    Returns:
    --------
    pd.DataFrame
        포트폴리오 DataFrame (columns: Ticker, Name, Weight)
        index: 1, 2, 3, ... (종목), '' (Cash 행)
    """
    if len(portfolio_dict) == 0:
        # 빈 포트폴리오: Cash 100%
        portfolio_df = pd.DataFrame([{
            'Ticker': 'Cash',
            'Name': '',
            'Weight': 1.0
        }])
        portfolio_df.index = ['']
        return portfolio_df

    ticker_to_name = dict(zip(tickers_info['Code'], tickers_info['Name']))

    # INVERSE와 일반 종목 분리
    inverse_weight = portfolio_dict.pop('INVERSE', 0.0)

    # 일반 종목들을 DataFrame으로 변환
    weights = []
    for ticker, weight in portfolio_dict.items():
        weights.append({
            'Ticker': f'S{ticker}',  # 'S' prefix 추가
            'Name': ticker_to_name.get(ticker, ''),
            'Weight': weight
        })

    # DataFrame 생성 및 Ticker 기준 오름차순 정렬
    portfolio = pd.DataFrame(weights)
    portfolio = portfolio.sort_values('Ticker')

    # 정렬 후 index 재설정 (1부터 시작)
    portfolio.index = range(1, len(portfolio) + 1)
    portfolio.index.name = 'No'

    # Cash 행 추가
    total_invested = portfolio['Weight'].sum() + inverse_weight
    cash_weight = 1.0 - total_invested
    cash_row = pd.DataFrame([{
        'Ticker': 'Cash',
        'Name': '',
        'Weight': cash_weight
    }], index=[''])
    portfolio = pd.concat([portfolio, cash_row])

    # 요약 정보 출력
    if verbose:
        n_invested = (portfolio['Weight'] > 0).sum() - 1  # Cash 제외
        print(f"      투자 종목: {n_invested}개 ({total_invested:.1%})")
        if inverse_weight > 0:
            print(f"      인버스 ETF: {inverse_weight:.1%}")
        print(f"      현금 보유: {cash_weight:.1%}")

    return portfolio


def calculate_portfolio_comparison(
    portfolio_current: pd.DataFrame,
    portfolio_1m_ago: pd.DataFrame
) -> pd.DataFrame:
    """
    현재 포트폴리오와 1달 전 포트폴리오 비교

    Parameters:
    -----------
    portfolio_current : pd.DataFrame
        현재 포트폴리오 (Ticker, Name, Weight 컬럼)
    portfolio_1m_ago : pd.DataFrame
        1달 전 포트폴리오 (Ticker, Name, Weight 컬럼)

    Returns:
    --------
    pd.DataFrame
        비교 리포트 (Ticker, Name, Weight_current, Weight_1m_ago, Weight_change, Status 컬럼)
    """
    # Cash 제외
    current_stocks = portfolio_current[portfolio_current['Ticker'] != 'Cash'].copy()
    prev_stocks = portfolio_1m_ago[portfolio_1m_ago['Ticker'] != 'Cash'].copy()

    # Ticker를 기준으로 merge (outer join)
    current_stocks = current_stocks.set_index('Ticker')
    prev_stocks = prev_stocks.set_index('Ticker')

    comparison = pd.DataFrame(index=sorted(set(current_stocks.index) | set(prev_stocks.index)))
    comparison['Name'] = current_stocks['Name'].combine_first(prev_stocks['Name'])
    comparison['Weight_current'] = current_stocks['Weight'].reindex(comparison.index).fillna(0.0)
    comparison['Weight_1m_ago'] = prev_stocks['Weight'].reindex(comparison.index).fillna(0.0)
    comparison['Weight_change'] = comparison['Weight_current'] - comparison['Weight_1m_ago']

    # Status 계산
    def get_status(row):
        if row['Weight_1m_ago'] == 0 and row['Weight_current'] > 0:
            return 'New'
        elif row['Weight_1m_ago'] > 0 and row['Weight_current'] == 0:
            return 'Removed'
        elif row['Weight_1m_ago'] == 0 and row['Weight_current'] == 0:
            return 'N/A'
        elif abs(row['Weight_change']) < 1e-6:
            return 'Unchanged'
        else:
            return 'Rebalanced'

    comparison['Status'] = comparison.apply(get_status, axis=1)

    # N/A 제외 및 정렬 (Status 우선, Ticker 차순)
    comparison = comparison[comparison['Status'] != 'N/A']
    status_order = {'New': 1, 'Removed': 2, 'Rebalanced': 3, 'Unchanged': 4}
    comparison['_sort_key'] = comparison['Status'].map(status_order)
    comparison = comparison.sort_values(by='_sort_key')
    comparison = comparison.sort_index()  # Ticker(index) 기준 2차 정렬
    comparison = comparison.drop(columns=['_sort_key'])

    # index를 Ticker 컬럼으로 복원
    comparison = comparison.reset_index()
    comparison = comparison.rename(columns={'index': 'Ticker'})

    # Cash 행 추가
    cash_current = portfolio_current[portfolio_current['Ticker'] == 'Cash']['Weight'].values[0]
    cash_1m_ago = portfolio_1m_ago[portfolio_1m_ago['Ticker'] == 'Cash']['Weight'].values[0]
    cash_change = cash_current - cash_1m_ago

    cash_row = pd.DataFrame([{
        'Ticker': 'Cash',
        'Name': '',
        'Weight_current': cash_current,
        'Weight_1m_ago': cash_1m_ago,
        'Weight_change': cash_change,
        'Status': 'Cash'
    }])

    comparison = pd.concat([comparison, cash_row], ignore_index=True)

    return comparison


# ============================================================
# Factory Pattern
# ============================================================

def create_strategy_selector(config: StrategyConfig) -> Callable:
    """
    전략 설정에 따라 포트폴리오 선택 함수를 생성하는 Factory 함수

    Parameters:
    -----------
    config : StrategyConfig
        전략 설정

    Returns:
    --------
    Callable
        (momentum, correlation) -> portfolio 형태의 선택 함수
    """
    def selector(momentum: pd.DataFrame, correlation: pd.DataFrame) -> Dict[str, float]:
        """전략 설정에 따른 포트폴리오 선택"""
        # 필터링
        tickers = apply_filters(
            momentum,
            correlation,
            config.momentum_ratio,
            config.rsquared_ratio,
            config.correlation_ratio
        )

        # 포트폴리오 구성
        portfolio = build_portfolio(
            tickers,
            momentum,
            use_inverse=config.use_inverse,
            use_macd_filter=config.use_macd_filter
        )

        return portfolio

    return selector
