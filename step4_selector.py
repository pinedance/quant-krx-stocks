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
from core.backtest import (
    StrategyConfig,
    calculate_signals_at_date,
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
# Helper Functions (step4 전용 로직만 유지)
# ============================================================


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
    print(f"      Tickers: {tickers_info.shape}")
    print(f"      closeM: {closeM.shape}")

    # 2. 현재 시점 포트폴리오 (최신 데이터 기준)
    print_progress(2, 5, f"현재 포트폴리오 계산 ({config.description})...")
    closeM_log = np.log(closeM)
    end_idx_current = len(closeM) - 1
    momentum_current, correlation_current = calculate_signals_at_date(
        closeM_log, closeM, end_idx_current, include_macd=config.use_macd_filter
    )

    # 필터링 (core.backtest.apply_filters 사용)
    from core.backtest import calculate_avg_momentum, calculate_avg_rsquared, calculate_marginal_means

    tickers_current = apply_filters(
        momentum_current,
        correlation_current,
        config.momentum_ratio,
        config.rsquared_ratio,
        config.correlation_ratio
    )

    # 선택된 종목 DataFrame 구성
    avg_mmt = calculate_avg_momentum(momentum_current)
    avg_rs = calculate_avg_rsquared(momentum_current)
    corr_matrix = correlation_current.drop('mean', axis=0, errors='ignore').drop('mean', axis=1, errors='ignore')
    marg_means = calculate_marginal_means(corr_matrix, tickers_current)

    selected_current = build_selected_dataframe(tickers_current, tickers_info, avg_mmt, avg_rs, marg_means)

    # 포트폴리오 구성
    portfolio_dict_current = build_portfolio(tickers_current, momentum_current, use_inverse=False, use_macd_filter=config.use_macd_filter)
    portfolio_current = format_portfolio_as_dataframe(portfolio_dict_current, tickers_info, verbose=True)

    # 3. 1달 전 시점 포트폴리오
    print_progress(3, 5, "1달 전 포트폴리오 계산...")
    if len(closeM) >= 2:
        end_idx_1m_ago = len(closeM) - 2
        momentum_1m_ago, correlation_1m_ago = calculate_signals_at_date(
            closeM_log, closeM, end_idx_1m_ago, include_macd=config.use_macd_filter
        )

        tickers_1m_ago = apply_filters(
            momentum_1m_ago,
            correlation_1m_ago,
            config.momentum_ratio,
            config.rsquared_ratio,
            config.correlation_ratio
        )

        avg_mmt_1m = calculate_avg_momentum(momentum_1m_ago)
        avg_rs_1m = calculate_avg_rsquared(momentum_1m_ago)
        corr_matrix_1m = correlation_1m_ago.drop('mean', axis=0, errors='ignore').drop('mean', axis=1, errors='ignore')
        marg_means_1m = calculate_marginal_means(corr_matrix_1m, tickers_1m_ago)

        selected_1m_ago = build_selected_dataframe(tickers_1m_ago, tickers_info, avg_mmt_1m, avg_rs_1m, marg_means_1m)

        portfolio_dict_1m_ago = build_portfolio(tickers_1m_ago, momentum_1m_ago, use_inverse=False, use_macd_filter=config.use_macd_filter)
        portfolio_1m_ago = format_portfolio_as_dataframe(portfolio_dict_1m_ago, tickers_info, verbose=True)
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
