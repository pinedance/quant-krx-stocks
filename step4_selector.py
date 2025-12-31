"""
STEP 4: 종목 선택 및 포트폴리오 구성
- Momentum/Performance 지표 기반 종목 필터링 및 순위 매기기
- 선택된 종목으로 포트폴리오 구성
"""

import pandas as pd
import numpy as np
from core.file import import_dataframe_from_json, export_with_message
from core.config import settings
from core.utils import print_step_header, print_progress, print_completion


def calculate_average_momentum(momentum):
    """
    평균 모멘텀 계산 (13612MR)

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 데이터

    Returns:
    --------
    pd.Series
        평균 모멘텀
    """
    return momentum['13612MR']


def calculate_average_rsquared(momentum):
    """
    평균 R-squared 계산: (1 + √RS3 + √RS6 + √RS12) / 4

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 데이터

    Returns:
    --------
    pd.Series
        평균 R-squared
    """
    return (1 + np.sqrt(momentum['RS3']) + np.sqrt(momentum['RS6']) +
            np.sqrt(momentum['RS12'])) / 4


def calculate_marginal_mean(correlation_matrix, tickers):
    """
    선택된 종목들 간의 marginal mean 계산

    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        전체 상관관계 행렬
    tickers : list
        선택된 종목 리스트

    Returns:
    --------
    pd.Series
        각 종목의 marginal mean (자기 자신 제외한 평균 상관계수)
    """
    # 선택된 종목들만 추출
    sub_corr = correlation_matrix.loc[tickers, tickers]

    # 각 종목의 marginal mean 계산 (자기 자신 제외)
    n = len(tickers)
    marginal_means = (sub_corr.sum(axis=1) - 1) / (n - 1)

    return marginal_means


def select_stocks_strategy1(momentum, correlation):
    """
    전략 1에 따른 종목 선택

    전략:
    1. 평균 모멘텀(13612MR) 상위 1/2
    2. 평균 R-squared 상위 1/2
    3. Marginal mean 하위 1/3
    최종: 전체의 1/12

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 데이터
    correlation : pd.DataFrame
        Correlation matrix

    Returns:
    --------
    pd.DataFrame
        선택된 종목 정보 (평균 모멘텀, 평균 R-squared, marginal mean 포함)
    """
    total_stocks = len(momentum)

    # Step 1: 평균 모멘텀 상위 1/2
    avg_momentum = calculate_average_momentum(momentum)
    step1_count = total_stocks // 2
    step1_tickers = avg_momentum.nlargest(step1_count).index.tolist()
    print(f"      Step 1: 평균 모멘텀 상위 1/2 → {len(step1_tickers)}개 종목")

    # Step 2: 평균 R-squared 상위 1/2
    avg_rsquared = calculate_average_rsquared(momentum)
    step2_candidates = avg_rsquared[step1_tickers]
    step2_count = len(step1_tickers) // 2
    step2_tickers = step2_candidates.nlargest(step2_count).index.tolist()
    print(f"      Step 2: 평균 R-squared 상위 1/2 → {len(step2_tickers)}개 종목")

    # Step 3: Marginal mean 하위 1/3 (상관관계가 낮은 종목)
    # correlation에서 'mean' 컬럼/행 제거 (step3에서 추가된 것)
    corr_matrix = correlation.drop('mean', axis=0, errors='ignore').drop('mean', axis=1, errors='ignore')
    marginal_means = calculate_marginal_mean(corr_matrix, step2_tickers)
    step3_count = len(step2_tickers) // 3
    step3_tickers = marginal_means.nsmallest(step3_count).index.tolist()
    print(f"      Step 3: Marginal mean 하위 1/3 → {len(step3_tickers)}개 종목")

    # 최종 선택 종목 정보
    selected = pd.DataFrame(index=step3_tickers)
    selected['avg_momentum'] = avg_momentum[step3_tickers]
    selected['avg_rsquared'] = avg_rsquared[step3_tickers]
    selected['marginal_mean'] = marginal_means[step3_tickers]

    print(f"      최종: {len(selected)}개 종목 (전체 {total_stocks}개의 {len(selected)/total_stocks:.1%})")

    return selected


def construct_portfolio(selected_stocks, tickers_info):
    """
    포트폴리오 구성 (동일 비중, 음수 모멘텀은 현금)

    전략:
    - 기본: 동일 비중 (1/N)
    - 평균 모멘텀 < 0: 해당 비중만큼 현금 보유

    Parameters:
    -----------
    selected_stocks : pd.DataFrame
        선택된 종목 리스트 (avg_momentum 포함)
    tickers_info : pd.DataFrame
        종목 정보 (Code, Name 포함)

    Returns:
    --------
    pd.DataFrame
        포트폴리오 구성 (No, Ticker, Name, Weight 컬럼)
    """
    n_stocks = len(selected_stocks)
    equal_weight = 1.0 / n_stocks if n_stocks > 0 else 0

    # 종목명 매핑
    ticker_to_name = dict(zip(tickers_info['Code'], tickers_info['Name']))

    # 투자 비중 계산
    weights = []
    for ticker in selected_stocks.index:
        avg_momentum = selected_stocks.loc[ticker, 'avg_momentum']
        weight = equal_weight if avg_momentum >= 0 else 0.0
        weights.append({
            'Ticker': ticker,
            'Name': ticker_to_name.get(ticker, ''),
            'Weight': weight
        })

    # DataFrame 생성
    portfolio = pd.DataFrame(weights)
    portfolio.index = range(1, len(portfolio) + 1)
    portfolio.index.name = 'No'

    # Cash 행 추가
    total_invested = portfolio['Weight'].sum()
    cash_weight = 1.0 - total_invested
    cash_row = pd.DataFrame([{
        'Ticker': 'Cash',
        'Name': '',
        'Weight': cash_weight
    }], index=[''])
    portfolio = pd.concat([portfolio, cash_row])

    # 요약 정보
    n_invested = (portfolio['Weight'] > 0).sum() - 1  # Cash 제외
    print(f"      투자 종목: {n_invested}개 ({total_invested:.1%})")
    print(f"      현금 보유: {cash_weight:.1%}")

    return portfolio


def main():
    print_step_header(4, "종목 선택 및 포트폴리오 구성")

    # 설정 로드
    market = settings.data.market
    list_dir = settings.output.list_dir.path
    signal_dir = settings.output.signal_dir.path
    portfolio_base_dir = settings.output.portfolio_dir.path
    strategy_name = "strategy1"  # 전략 1
    output_dir = f"{portfolio_base_dir}/{strategy_name}"

    # 1. 종목 리스트 및 Signal 데이터 로드
    print_progress(1, 3, "데이터 로드...")
    tickers_info = import_dataframe_from_json(f'{list_dir}/{market}_list.json')
    momentum = import_dataframe_from_json(f'{signal_dir}/momentum.json')
    correlation = import_dataframe_from_json(f'{signal_dir}/correlation.json')
    print(f"      Tickers: {tickers_info.shape}")
    print(f"      Momentum: {momentum.shape}")
    print(f"      Correlation: {correlation.shape}")

    # 2. 종목 선택 (전략 1)
    print_progress(2, 3, "종목 선택 (전략 1)...")
    selected = select_stocks_strategy1(momentum, correlation)

    # 3. 포트폴리오 구성
    print_progress(3, 3, "포트폴리오 구성...")
    portfolio = construct_portfolio(selected, tickers_info)

    # 4. 저장
    print(f"\n파일 저장 (HTML, TSV, JSON) → {output_dir}/...")
    export_with_message(selected, f'{output_dir}/selected', 'Selected Stocks')
    export_with_message(portfolio, f'{output_dir}/portfolio', 'Portfolio Composition')

    print_completion(4)


if __name__ == "__main__":
    main()
