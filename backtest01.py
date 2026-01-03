"""
BACKTEST: Strategy 1 백테스트
- 시총 상위 300개 종목(조건: 1년 데이터 존재) | 13612MR 상위 1/2 종목 | mean-R2 상위 1/2종목 | correlation marginal mean 하위 1/3 종목
    - 13612MR: ( 1M Return + 3M Return + 6M Return + 12M Return) / 4
    - mean-R2: ( 1 + sqrt(RS3) + sqrt(RS6) + sqrt(RS12) ) / 4
- 위 종목에 1/N 비중으로 투자. 단, 13612MR이 0보다 작은 종목에는 해당 비중만큼 현금 보유
- 매월 1일 리밸런싱 (현재가: 지난달 종가)
- 벤치마크 ETF Ticker: 069500 (KOSPI200)
"""

"""
BACKTEST: Strategy 2 백테스트
- Strategy 1 로직 변형 (다른 조건 동일)
- 13612MR이 0보다 작은 종목에는 해당 비중의 1/4 인버스 보유, 3/4 현금 보유
    - 인버스 ETF Ticker: 114800
"""

"""
BACKTEST: Strategy 3 백테스트
- Strategy 1 로직 변형 (다른 조건 동일)
- 시총 상위 300개 종목(조건: 1년 데이터 존재) | 13612MR 상위 1/2 종목 | mean-R2 상위 1/2종목 | correlation marginal mean 하위 1/4 종목
"""


"""
BACKTEST: Strategy 4 백테스트
- Strategy 1 로직 변형 (다른 조건 동일)
- 시총 상위 300개 종목(조건: 1년 데이터 존재) | 13612MR 상위 1/3 종목 | mean-R2 상위 1/3종목 | correlation marginal mean 하위 1/3 종목
"""

import pandas as pd
import numpy as np
from core.file import import_dataframe_from_json
from core.config import settings
from core.utils import print_step_header, print_progress, print_completion, ensure_directory
from core.backtest import BacktestRunner

SUBDIR = "backtest01"  # 현재 자기 자신 python file name


# ============================================================
# Strategy 포트폴리오 선택 함수
# ============================================================

def select_portfolio_strategy1(momentum, correlation):
    """
    Strategy 1: 13612MR 상위 1/2 | mean-R2 상위 1/2 | correlation marginal mean 하위 1/3
    """
    momentum = momentum.dropna(subset=['13612MR', 'RS3', 'RS6', 'RS12'])
    if len(momentum) == 0:
        return {}

    total_stocks = len(momentum)

    # Step 1: 평균 모멘텀 상위 1/2
    avg_momentum = momentum['13612MR']
    step1_count = max(1, total_stocks // 2)
    step1_tickers = avg_momentum.nlargest(step1_count).index.tolist()

    # Step 2: 평균 R-squared 상위 1/2
    avg_rsquared = (1 + np.sqrt(momentum['RS3']) +
                    np.sqrt(momentum['RS6']) +
                    np.sqrt(momentum['RS12'])) / 4
    step2_candidates = avg_rsquared[step1_tickers]
    step2_count = max(1, len(step1_tickers) // 2)
    step2_tickers = step2_candidates.nlargest(step2_count).index.tolist()

    # Step 3: Marginal mean 하위 1/3
    corr_matrix = correlation.drop('mean', axis=0, errors='ignore').drop('mean', axis=1, errors='ignore')
    available_tickers = [t for t in step2_tickers if t in corr_matrix.index]
    if len(available_tickers) == 0:
        return {}

    sub_corr = corr_matrix.loc[available_tickers, available_tickers]
    n = len(available_tickers)

    if n > 1:
        marginal_means = (sub_corr.sum(axis=1) - 1) / (n - 1)
        step3_count = max(1, n // 3)
        step3_tickers = marginal_means.nsmallest(step3_count).index.tolist()
    else:
        step3_tickers = available_tickers

    # 포트폴리오 구성 (동일 비중, 음수 모멘텀은 현금)
    n_stocks = len(step3_tickers)
    equal_weight = 1.0 / n_stocks if n_stocks > 0 else 0

    portfolio = {}
    for ticker in step3_tickers:
        if momentum.loc[ticker, '13612MR'] >= 0:
            portfolio[ticker] = equal_weight

    return portfolio


def select_portfolio_strategy2(momentum, correlation):
    """
    Strategy 2: Strategy 1 + 음수 모멘텀 종목에 1/4 인버스 보유
    """
    momentum = momentum.dropna(subset=['13612MR', 'RS3', 'RS6', 'RS12'])
    if len(momentum) == 0:
        return {}

    total_stocks = len(momentum)

    # Step 1: 평균 모멘텀 상위 1/2
    avg_momentum = momentum['13612MR']
    step1_count = max(1, total_stocks // 2)
    step1_tickers = avg_momentum.nlargest(step1_count).index.tolist()

    # Step 2: 평균 R-squared 상위 1/2
    avg_rsquared = (1 + np.sqrt(momentum['RS3']) +
                    np.sqrt(momentum['RS6']) +
                    np.sqrt(momentum['RS12'])) / 4
    step2_candidates = avg_rsquared[step1_tickers]
    step2_count = max(1, len(step1_tickers) // 2)
    step2_tickers = step2_candidates.nlargest(step2_count).index.tolist()

    # Step 3: Marginal mean 하위 1/3
    corr_matrix = correlation.drop('mean', axis=0, errors='ignore').drop('mean', axis=1, errors='ignore')
    available_tickers = [t for t in step2_tickers if t in corr_matrix.index]
    if len(available_tickers) == 0:
        return {}

    sub_corr = corr_matrix.loc[available_tickers, available_tickers]
    n = len(available_tickers)

    if n > 1:
        marginal_means = (sub_corr.sum(axis=1) - 1) / (n - 1)
        step3_count = max(1, n // 3)
        step3_tickers = marginal_means.nsmallest(step3_count).index.tolist()
    else:
        step3_tickers = available_tickers

    # 포트폴리오 구성 (동일 비중, 음수 모멘텀은 1/4 인버스)
    n_stocks = len(step3_tickers)
    equal_weight = 1.0 / n_stocks if n_stocks > 0 else 0

    portfolio = {}
    inverse_weight = 0.0

    for ticker in step3_tickers:
        if momentum.loc[ticker, '13612MR'] >= 0:
            portfolio[ticker] = equal_weight
        else:
            inverse_weight += equal_weight / 4

    if inverse_weight > 0:
        portfolio['INVERSE'] = inverse_weight

    return portfolio


def select_portfolio_strategy3(momentum, correlation):
    """
    Strategy 3: 13612MR 상위 1/2 | mean-R2 상위 1/2 | correlation marginal mean 하위 1/4
    """
    momentum = momentum.dropna(subset=['13612MR', 'RS3', 'RS6', 'RS12'])
    if len(momentum) == 0:
        return {}

    total_stocks = len(momentum)

    # Step 1: 평균 모멘텀 상위 1/2
    avg_momentum = momentum['13612MR']
    step1_count = max(1, total_stocks // 2)
    step1_tickers = avg_momentum.nlargest(step1_count).index.tolist()

    # Step 2: 평균 R-squared 상위 1/2
    avg_rsquared = (1 + np.sqrt(momentum['RS3']) +
                    np.sqrt(momentum['RS6']) +
                    np.sqrt(momentum['RS12'])) / 4
    step2_candidates = avg_rsquared[step1_tickers]
    step2_count = max(1, len(step1_tickers) // 2)
    step2_tickers = step2_candidates.nlargest(step2_count).index.tolist()

    # Step 3: Marginal mean 하위 1/4
    corr_matrix = correlation.drop('mean', axis=0, errors='ignore').drop('mean', axis=1, errors='ignore')
    available_tickers = [t for t in step2_tickers if t in corr_matrix.index]
    if len(available_tickers) == 0:
        return {}

    sub_corr = corr_matrix.loc[available_tickers, available_tickers]
    n = len(available_tickers)

    if n > 1:
        marginal_means = (sub_corr.sum(axis=1) - 1) / (n - 1)
        step3_count = max(1, n // 4)
        step3_tickers = marginal_means.nsmallest(step3_count).index.tolist()
    else:
        step3_tickers = available_tickers

    # 포트폴리오 구성
    n_stocks = len(step3_tickers)
    equal_weight = 1.0 / n_stocks if n_stocks > 0 else 0

    portfolio = {}
    for ticker in step3_tickers:
        if momentum.loc[ticker, '13612MR'] >= 0:
            portfolio[ticker] = equal_weight

    return portfolio


def select_portfolio_strategy4(momentum, correlation):
    """
    Strategy 4: 13612MR 상위 1/3 | mean-R2 상위 1/3 | correlation marginal mean 하위 1/3
    """
    momentum = momentum.dropna(subset=['13612MR', 'RS3', 'RS6', 'RS12'])
    if len(momentum) == 0:
        return {}

    total_stocks = len(momentum)

    # Step 1: 평균 모멘텀 상위 1/3
    avg_momentum = momentum['13612MR']
    step1_count = max(1, total_stocks // 3)
    step1_tickers = avg_momentum.nlargest(step1_count).index.tolist()

    # Step 2: 평균 R-squared 상위 1/3
    avg_rsquared = (1 + np.sqrt(momentum['RS3']) +
                    np.sqrt(momentum['RS6']) +
                    np.sqrt(momentum['RS12'])) / 4
    step2_candidates = avg_rsquared[step1_tickers]
    step2_count = max(1, len(step1_tickers) // 3)
    step2_tickers = step2_candidates.nlargest(step2_count).index.tolist()

    # Step 3: Marginal mean 하위 1/3
    corr_matrix = correlation.drop('mean', axis=0, errors='ignore').drop('mean', axis=1, errors='ignore')
    available_tickers = [t for t in step2_tickers if t in corr_matrix.index]
    if len(available_tickers) == 0:
        return {}

    sub_corr = corr_matrix.loc[available_tickers, available_tickers]
    n = len(available_tickers)

    if n > 1:
        marginal_means = (sub_corr.sum(axis=1) - 1) / (n - 1)
        step3_count = max(1, n // 3)
        step3_tickers = marginal_means.nsmallest(step3_count).index.tolist()
    else:
        step3_tickers = available_tickers

    # 포트폴리오 구성
    n_stocks = len(step3_tickers)
    equal_weight = 1.0 / n_stocks if n_stocks > 0 else 0

    portfolio = {}
    for ticker in step3_tickers:
        if momentum.loc[ticker, '13612MR'] >= 0:
            portfolio[ticker] = equal_weight

    return portfolio


# ============================================================
# Main
# ============================================================

def main():
    print_step_header(0, "Strategy 1-4 백테스트")

    # 설정 로드
    price_dir = settings.output.price_dir.path
    backtest_base_dir = settings.output.backtest_dir.path
    output_dir = f"{backtest_base_dir}/{SUBDIR}"
    end_date = settings.backtest.end_date if hasattr(settings, 'backtest') else None
    ensure_directory(output_dir)

    # 1. 가격 데이터 로드
    print_progress(1, 4, "가격 데이터 로드...")
    closeM = import_dataframe_from_json(f'{price_dir}/closeM.json')
    closeM.index = pd.to_datetime(closeM.index)

    # 전체 기간 동안 NaN이 없는 종목만 선택 (최대 300개)
    closeM_complete = closeM.dropna(axis=1, how='any')
    n_stocks = min(300, len(closeM_complete.columns))
    closeM = closeM_complete.iloc[:, :n_stocks]

    print(f"      데이터 기간: {closeM.index[0]} ~ {closeM.index[-1]}")
    print(f"      종목 수: {len(closeM.columns)}개")
    print(f"      총 {len(closeM)}개월")

    if end_date:
        print(f"      백테스트 종료일 설정: {end_date}")

    # 2. 백테스트 러너 초기화
    print_progress(2, 4, "벤치마크 및 인버스 ETF 데이터 로드...")
    runner = BacktestRunner(closeM, output_dir, end_date)
    runner.load_benchmark('069500')
    runner.load_inverse_etf('114800')

    # 3. 전략별 백테스트 실행
    print_progress(3, 4, "백테스트 실행 중...")
    runner.add_strategy('strategy1', select_portfolio_strategy1)
    runner.add_strategy('strategy2', select_portfolio_strategy2, use_inverse=True)
    runner.add_strategy('strategy3', select_portfolio_strategy3)
    runner.add_strategy('strategy4', select_portfolio_strategy4)

    # 4. 결과 저장 및 출력
    print_progress(4, 4, "결과 저장 중...")
    metrics_df = runner.save_results()

    print("\n" + "="*70)
    print("백테스트 결과")
    print("="*70)
    print(metrics_df.to_string())

    print_completion(0)


if __name__ == "__main__":
    main()
