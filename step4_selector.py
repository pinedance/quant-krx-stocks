"""
STEP 4: 종목 선택 및 포트폴리오 구성
- Momentum/Performance 지표 기반 종목 필터링 및 순위 매기기
- 선택된 종목으로 포트폴리오 구성
"""

import pandas as pd
from dataclasses import dataclass
from core.file import import_dataframe_from_json, export_with_message, export_dataframe_to_datatable
from core.config import settings
from core.utils import print_step_header, print_progress, print_completion
from core.backtest import calculate_avg_momentum, calculate_avg_rsquared, calculate_marginal_means


# ============================================================
# Selection Configuration Pattern
# ============================================================

@dataclass
class SelectionConfig:
    """
    종목 선택 전략 설정

    Attributes:
    -----------
    name : str
        전략 이름 (예: 'strategy1')
    momentum_ratio : float
        모멘텀 상위 비율 (0.5 = 상위 1/2)
    rsquared_ratio : float
        R-squared 상위 비율
    correlation_ratio : float
        Correlation 하위 비율 (낮을수록 분산 효과)
    description : str
        전략 설명
    """
    name: str
    momentum_ratio: float
    rsquared_ratio: float
    correlation_ratio: float
    description: str = ""


# 전략 설정 정의
SELECTION_STRATEGIES = [
    SelectionConfig(
        name="strategy1",
        momentum_ratio=0.5,
        rsquared_ratio=0.5,
        correlation_ratio=0.33,
        description="Base - 모멘텀 1/2 | R² 1/2 | 상관관계 1/3"
    ),
    # 추가 전략 정의 예시:
    # SelectionConfig(
    #     name="strategy2",
    #     momentum_ratio=0.5,
    #     rsquared_ratio=0.33,
    #     correlation_ratio=0.25,
    #     description="Quality Focus - R² 강화"
    # ),
]


# ============================================================
# 공통 로직
# ============================================================

def apply_selection_filters(
    momentum: pd.DataFrame,
    correlation: pd.DataFrame,
    momentum_ratio: float,
    rsquared_ratio: float,
    correlation_ratio: float
) -> tuple:
    """
    3단계 필터링을 수행하여 종목 선택

    Parameters:
    -----------
    momentum : pd.DataFrame
        모멘텀 데이터
    correlation : pd.DataFrame
        상관관계 행렬
    momentum_ratio : float
        모멘텀 상위 비율
    rsquared_ratio : float
        R-squared 상위 비율
    correlation_ratio : float
        Correlation 하위 비율

    Returns:
    --------
    tuple
        (selected_tickers, avg_momentum, avg_rsquared, marginal_means)
    """
    total_stocks = len(momentum)

    # Step 1: 평균 모멘텀 필터링
    avg_momentum = calculate_avg_momentum(momentum)
    step1_count = int(total_stocks * momentum_ratio)
    step1_tickers = avg_momentum.nlargest(step1_count).index.tolist()
    print(f"      Step 1: 평균 모멘텀 상위 {momentum_ratio:.0%} → {len(step1_tickers)}개 종목")

    # Step 2: 평균 R-squared 필터링
    avg_rsquared = calculate_avg_rsquared(momentum)
    step2_candidates = avg_rsquared[step1_tickers]
    step2_count = int(len(step1_tickers) * rsquared_ratio)
    step2_tickers = step2_candidates.nlargest(step2_count).index.tolist()
    print(f"      Step 2: 평균 R-squared 상위 {rsquared_ratio:.0%} → {len(step2_tickers)}개 종목")

    # Step 3: Marginal mean 필터링
    corr_matrix = correlation.drop('mean', axis=0, errors='ignore').drop('mean', axis=1, errors='ignore')
    marginal_means = calculate_marginal_means(corr_matrix, step2_tickers)
    step3_count = int(len(step2_tickers) * correlation_ratio)
    step3_tickers = marginal_means.nsmallest(step3_count).index.tolist()
    print(f"      Step 3: Marginal mean 하위 {correlation_ratio:.0%} → {len(step3_tickers)}개 종목")

    print(f"      최종: {len(step3_tickers)}개 종목 (전체 {total_stocks}개의 {len(step3_tickers)/total_stocks:.1%})")

    return step3_tickers, avg_momentum, avg_rsquared, marginal_means


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
        전체 종목 정보
    avg_momentum : pd.Series
        평균 모멘텀
    avg_rsquared : pd.Series
        평균 R-squared
    marginal_means : pd.Series
        Marginal means

    Returns:
    --------
    pd.DataFrame
        선택된 종목 정보
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


def create_selection_strategy(config: SelectionConfig):
    """
    전략 설정에 따라 종목 선택 함수를 생성하는 Factory 함수

    Parameters:
    -----------
    config : SelectionConfig
        전략 설정

    Returns:
    --------
    function
        종목 선택 함수
    """
    def selector(momentum: pd.DataFrame, correlation: pd.DataFrame, tickers_info: pd.DataFrame) -> pd.DataFrame:
        """전략 설정에 따른 종목 선택"""
        # 필터링
        tickers, avg_mmt, avg_rs, marg_means = apply_selection_filters(
            momentum,
            correlation,
            config.momentum_ratio,
            config.rsquared_ratio,
            config.correlation_ratio
        )

        # DataFrame 구성
        selected = build_selected_dataframe(
            tickers,
            tickers_info,
            avg_mmt,
            avg_rs,
            marg_means
        )

        return selected

    return selector


# ============================================================
# 기존 함수 (하위 호환성 유지)
# ============================================================


def select_stocks_strategy1(momentum, correlation, tickers_info):
    """
    전략 1에 따른 종목 선택 (레거시 함수 - 하위 호환성 유지)

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
    tickers_info : pd.DataFrame
        종목 정보 (Code, Name 포함)

    Returns:
    --------
    pd.DataFrame
        선택된 종목 정보 (Ticker, Name, 평균 모멘텀, 평균 R-squared, marginal mean 포함)
    """
    # Strategy Config 사용
    config = SELECTION_STRATEGIES[0]  # strategy1
    selector = create_selection_strategy(config)
    return selector(momentum, correlation, tickers_info)


def construct_portfolio(selected_stocks):
    """
    포트폴리오 구성 (동일 비중, 음수 모멘텀은 현금)

    전략:
    - 기본: 동일 비중 (1/N)
    - 평균 모멘텀 < 0: 해당 비중만큼 현금 보유

    Parameters:
    -----------
    selected_stocks : pd.DataFrame
        선택된 종목 리스트 (Ticker index, Name, avg_momentum 포함)

    Returns:
    --------
    pd.DataFrame
        포트폴리오 구성 (No, Ticker, Name, Weight 컬럼)
    """
    n_stocks = len(selected_stocks)
    equal_weight = 1.0 / n_stocks if n_stocks > 0 else 0

    # 투자 비중 계산
    weights = []
    for ticker in selected_stocks.index:
        avg_momentum = selected_stocks.loc[ticker, 'avg_momentum']
        name = selected_stocks.loc[ticker, 'Name']
        weight = equal_weight if avg_momentum >= 0 else 0.0
        weights.append({
            'Ticker': ticker,  # 이미 'S' prefix 포함
            'Name': name,
            'Weight': weight
        })

    # DataFrame 생성 및 Ticker 기준 오름차순 정렬
    portfolio = pd.DataFrame(weights)
    portfolio = portfolio.sort_values('Ticker')

    # 정렬 후 index 재설정
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
    market = settings.stocks.list.market
    list_dir = settings.output.list_dir.path
    signal_dir = settings.output.signal_dir.path
    portfolio_base_dir = settings.output.portfolio_dir.path
    strategy_name = "strategy1"  # 전략 1
    output_dir = f"{portfolio_base_dir}/{strategy_name}"

    # 1. 종목 리스트 및 Signal 데이터 로드
    print_progress(1, 3, "데이터 로드...")
    tickers_info = import_dataframe_from_json(f'{list_dir}/{market}.json')
    momentum = import_dataframe_from_json(f'{signal_dir}/momentum.json')
    correlation = import_dataframe_from_json(f'{signal_dir}/correlation.json')
    print(f"      Tickers: {tickers_info.shape}")
    print(f"      Momentum: {momentum.shape}")
    print(f"      Correlation: {correlation.shape}")

    # 2. 종목 선택 (전략 1)
    print_progress(2, 3, "종목 선택 (전략 1)...")
    selected = select_stocks_strategy1(momentum, correlation, tickers_info)

    # 3. 포트폴리오 구성
    print_progress(3, 3, "포트폴리오 구성...")
    portfolio = construct_portfolio(selected)

    # 4. 저장
    print(f"\n파일 저장 (HTML, TSV, JSON) → {output_dir}/...")
    export_with_message(selected, f'{output_dir}/selected', 'Selected Stocks')
    export_with_message(portfolio, f'{output_dir}/portfolio', 'Portfolio Composition')

    # DataTables 인터랙티브 버전 추가
    print(f"\n인터랙티브 테이블 생성 (DataTables)...")
    export_dataframe_to_datatable(selected, f'{output_dir}/selected', 'Selected Stocks - Interactive Table')

    print_completion(4)


if __name__ == "__main__":
    main()
