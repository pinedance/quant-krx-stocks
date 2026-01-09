"""
STEP 4: 종목 선택 및 포트폴리오 구성
- Momentum/Performance 지표 기반 종목 필터링 및 순위 매기기
- 선택된 종목으로 포트폴리오 구성
"""

import pandas as pd
import numpy as np
from core.file import import_dataframe_from_json, export_with_message, export_dataframe_to_datatable
from core.config import settings
from core.utils import print_step_header, print_progress, print_completion
from core.signals import calculate_signals_at_date
from core.strategy import (
    StrategyConfig,
    calculate_avg_momentum, calculate_avg_rsquared, calculate_marginal_means,
    apply_filters,
    build_selected_dataframe,
    build_portfolio,
    format_portfolio_as_dataframe,
    calculate_portfolio_comparison
)

# ============================================================
# Selection Configuration Pattern
# ============================================================

# 전략 설정 정의 (core.backtest.StrategyConfig 사용)
SELECTION_STRATEGIES = [
    StrategyConfig(
        name="main",   # S234MACD
        momentum_ratio=1/2,
        rsquared_ratio=1/3,
        correlation_ratio=1/4,
        use_macd_filter=True,
        description="S234MACD - 모멘텀 1/2 | R² 1/3 | 상관관계 1/4 | MACD 필터"
    ),
]


# ============================================================
# Factory Pattern (backtest와 일관성 유지)
# ============================================================


def create_portfolio_builder(config: StrategyConfig, tickers_info: pd.DataFrame):
    """
    포트폴리오 빌더 함수를 생성하는 Factory 함수

    backtest01.py의 create_strategy_selector() 패턴과 일관성 유지

    Parameters:
    -----------
    config : StrategyConfig
        전략 설정
    tickers_info : pd.DataFrame
        종목 정보 (Code, Name 컬럼)

    Returns:
    --------
    Callable
        builder(closeM, closeM_log, end_idx, verbose) -> (selected_df, portfolio_df)
    """
    def builder(
        closeM: pd.DataFrame,
        closeM_log: pd.DataFrame,
        end_idx: int,
        verbose: bool = True
    ) -> tuple:
        """
        특정 시점의 포트폴리오 계산

        Parameters:
        -----------
        closeM : pd.DataFrame
            월별 종가 데이터
        closeM_log : pd.DataFrame
            로그 변환된 월별 종가
        end_idx : int
            계산 기준 인덱스 (음수 가능: -1=최신, -2=1달 전)
        verbose : bool
            출력 여부

        Returns:
        --------
        tuple
            (selected_df, portfolio_df)
        """
        # 인덱스 정규화
        if end_idx < 0:
            end_idx = len(closeM) + end_idx

        # 1. 시그널 계산
        momentum, correlation = calculate_signals_at_date(
            closeM_log, closeM, end_idx, include_macd=config.use_macd_filter
        )

        # 2. 필터링 (backtest의 apply_filters와 동일)
        tickers = apply_filters(
            momentum,
            correlation,
            config.momentum_ratio,
            config.rsquared_ratio,
            config.correlation_ratio
        )

        # 3. 선택된 종목 DataFrame 구성
        avg_mmt = calculate_avg_momentum(momentum)
        avg_rs = calculate_avg_rsquared(momentum)
        corr_matrix = correlation.drop('mean', axis=0, errors='ignore').drop('mean', axis=1, errors='ignore')
        marg_means = calculate_marginal_means(corr_matrix, tickers)

        selected = build_selected_dataframe(tickers, tickers_info, avg_mmt, avg_rs, marg_means)

        # 4. 포트폴리오 구성 (backtest의 build_portfolio와 동일)
        portfolio_dict = build_portfolio(
            tickers, momentum,
            use_inverse=config.use_inverse,
            use_macd_filter=config.use_macd_filter
        )
        portfolio = format_portfolio_as_dataframe(portfolio_dict, tickers_info, verbose=verbose)

        return selected, portfolio

    return builder


def main():
    print_step_header(4, "종목 선택 및 포트폴리오 구성")

    # 설정 로드
    market = settings.stocks.list.market
    list_dir = settings.output.list_dir.path
    price_dir = settings.output.price_dir.path
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
    closeM_log = np.log(closeM)
    print(f"      Tickers: {tickers_info.shape}")
    print(f"      closeM: {closeM.shape}")

    # 2. Factory 패턴으로 빌더 생성 (backtest01.py와 동일 패턴)
    print_progress(2, 5, f"포트폴리오 빌더 생성 ({config.description})...")
    builder = create_portfolio_builder(config, tickers_info)

    # 3. 현재 시점 포트폴리오
    print_progress(3, 5, "현재 포트폴리오 계산...")
    selected_current, portfolio_current = builder(closeM, closeM_log, -1, verbose=True)

    # 4. 1달 전 시점 포트폴리오
    print_progress(4, 5, "1달 전 포트폴리오 계산...")
    if len(closeM) >= 2:
        selected_1m_ago, portfolio_1m_ago = builder(closeM, closeM_log, -2, verbose=True)
    else:
        print("      경고: 데이터 부족으로 1달 전 포트폴리오를 계산할 수 없습니다.")
        selected_1m_ago = None
        portfolio_1m_ago = None

    # 5. 포트폴리오 비교
    print_progress(5, 5, "포트폴리오 비교 및 저장...")
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

    # 6. 저장
    print(f"\n파일 저장 (HTML, TSV, JSON) → {output_dir}/...")

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
    selected_dt_path = export_dataframe_to_datatable(selected_current, f'{output_dir}/selected', 'Selected Stocks (Current) - Interactive Table')
    print(f"  ✓ {selected_dt_path}")
    if comparison is not None:
        comparison_dt_path = export_dataframe_to_datatable(comparison, f'{output_dir}/portfolio_comparison', 'Portfolio Comparison - Interactive Table')
        print(f"  ✓ {comparison_dt_path}")

    print_completion(4)


if __name__ == "__main__":
    main()
