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
from core.backtest import calculate_avg_momentum, calculate_avg_rsquared, calculate_marginal_means, calculate_signals_at_date
import numpy as np


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
    use_macd_filter : bool
        MACD 오실레이터 필터 사용 여부
    description : str
        전략 설명
    """
    name: str
    momentum_ratio: float
    rsquared_ratio: float
    correlation_ratio: float
    use_macd_filter: bool = False
    description: str = ""


# 전략 설정 정의
SELECTION_STRATEGIES = [
    SelectionConfig(
        name="main",   #S234MACD
        momentum_ratio=1/2,
        rsquared_ratio=1/3,
        correlation_ratio=1/4,
        use_macd_filter=True,
        description="S234MACD - 모멘텀 1/2 | R² 1/3 | 상관관계 1/4 | MACD 필터"
    ),
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


def construct_portfolio(selected_stocks, momentum=None, use_macd_filter=False, verbose=True):
    """
    포트폴리오 구성 (동일 비중, 음수 모멘텀은 현금)

    전략:
    - 기본: 동일 비중 (1/N)
    - 평균 모멘텀 < 0: 해당 비중만큼 현금 보유
    - MACD 필터: MACD Histogram < 0인 경우도 현금 보유

    Parameters:
    -----------
    selected_stocks : pd.DataFrame
        선택된 종목 리스트 (Ticker index, Name, avg_momentum 포함)
    momentum : pd.DataFrame, optional
        모멘텀 데이터 (MACD_Histogram 컬럼 포함, MACD 필터 사용 시 필요)
    use_macd_filter : bool
        MACD 오실레이터 필터 사용 여부
    verbose : bool
        요약 정보 출력 여부

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

        # Ticker에서 'S' prefix 제거하여 실제 종목코드 추출
        ticker_code = ticker[1:] if ticker.startswith('S') else ticker

        # 기본 필터: 평균 모멘텀 체크
        weight = equal_weight if avg_momentum >= 0 else 0.0

        # MACD 필터 적용
        if use_macd_filter and weight > 0 and momentum is not None:
            if ticker_code in momentum.index:
                macd_hist = momentum.loc[ticker_code, 'MACD_Histogram']
                if pd.notna(macd_hist) and macd_hist < 0:
                    weight = 0.0

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
    if verbose:
        n_invested = (portfolio['Weight'] > 0).sum() - 1  # Cash 제외
        print(f"      투자 종목: {n_invested}개 ({total_invested:.1%})")
        print(f"      현금 보유: {cash_weight:.1%}")

    return portfolio


def calculate_portfolio_comparison(portfolio_current, portfolio_1m_ago):
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


def main():
    print_step_header(4, "종목 선택 및 포트폴리오 구성")

    # 설정 로드
    market = settings.stocks.list.market
    list_dir = settings.output.list_dir.path
    price_dir = settings.output.price_dir.path
    signal_dir = settings.output.signal_dir.path
    portfolio_base_dir = settings.output.portfolio_dir.path

    # 전략 설정
    config = SELECTION_STRATEGIES[0]  # main (S234MACD)
    strategy_name = config.name
    output_dir = f"{portfolio_base_dir}/{strategy_name}"

    # 1. 데이터 로드
    print_progress(1, 5, "데이터 로드...")
    tickers_info = import_dataframe_from_json(f'{list_dir}/{market}.json')
    closeM = import_dataframe_from_json(f'{price_dir}/closeM.json')
    closeM.index = pd.to_datetime(closeM.index)
    print(f"      Tickers: {tickers_info.shape}")
    print(f"      closeM: {closeM.shape}")

    # 2. 현재 시점 포트폴리오 (최신 데이터 기준)
    print_progress(2, 5, f"현재 포트폴리오 계산 ({config.description})...")
    closeM_log = np.log(closeM)
    end_idx_current = len(closeM) - 1
    momentum_current, correlation_current = calculate_signals_at_date(
        closeM_log, closeM, end_idx_current, include_macd=config.use_macd_filter
    )

    selector = create_selection_strategy(config)
    selected_current = selector(momentum_current, correlation_current, tickers_info)
    portfolio_current = construct_portfolio(selected_current, momentum_current, config.use_macd_filter, verbose=True)

    # 3. 1달 전 시점 포트폴리오
    print_progress(3, 5, "1달 전 포트폴리오 계산...")
    if len(closeM) >= 2:
        end_idx_1m_ago = len(closeM) - 2
        momentum_1m_ago, correlation_1m_ago = calculate_signals_at_date(
            closeM_log, closeM, end_idx_1m_ago, include_macd=config.use_macd_filter
        )

        selected_1m_ago = selector(momentum_1m_ago, correlation_1m_ago, tickers_info)
        portfolio_1m_ago = construct_portfolio(selected_1m_ago, momentum_1m_ago, config.use_macd_filter, verbose=True)
    else:
        print("      경고: 데이터 부족으로 1달 전 포트폴리오를 계산할 수 없습니다.")
        selected_1m_ago = None
        portfolio_1m_ago = None

    # 4. 포트폴리오 비교
    print_progress(4, 5, "포트폴리오 비교 리포트 생성...")
    if portfolio_1m_ago is not None:
        comparison = calculate_portfolio_comparison(portfolio_current, portfolio_1m_ago)

        # 요약 통계
        n_new = (comparison['Status'] == 'New').sum()
        n_removed = (comparison['Status'] == 'Removed').sum()
        n_rebalanced = (comparison['Status'] == 'Rebalanced').sum()
        n_unchanged = (comparison['Status'] == 'Unchanged').sum()

        print(f"      신규 편입: {n_new}개")
        print(f"      제외: {n_removed}개")
        print(f"      비중 조정: {n_rebalanced}개")
        print(f"      변동 없음: {n_unchanged}개")
    else:
        comparison = None

    # 5. 저장
    print_progress(5, 5, f"파일 저장 (HTML, TSV, JSON) → {output_dir}/...")

    # 현재 포트폴리오
    export_with_message(selected_current, f'{output_dir}/selected', 'Selected Stocks (Current)')
    export_with_message(portfolio_current, f'{output_dir}/portfolio', 'Portfolio Composition (Current)')

    # 1달 전 포트폴리오
    if selected_1m_ago is not None and portfolio_1m_ago is not None:
        export_with_message(selected_1m_ago, f'{output_dir}/selected_1m_ago', 'Selected Stocks (1 Month Ago)')
        export_with_message(portfolio_1m_ago, f'{output_dir}/portfolio_1m_ago', 'Portfolio Composition (1 Month Ago)')

    # 비교 리포트
    if comparison is not None:
        export_with_message(comparison, f'{output_dir}/portfolio_comparison', 'Portfolio Comparison (Current vs 1M Ago)')

    # DataTables 인터랙티브 버전 추가
    print("\n인터랙티브 테이블 생성 (DataTables)...")
    export_dataframe_to_datatable(selected_current, f'{output_dir}/selected', 'Selected Stocks (Current) - Interactive Table')
    if comparison is not None:
        export_dataframe_to_datatable(comparison, f'{output_dir}/portfolio_comparison', 'Portfolio Comparison - Interactive Table')

    print_completion(4)


if __name__ == "__main__":
    main()
