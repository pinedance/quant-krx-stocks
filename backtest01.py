"""
KRX300 모멘텀 전략 백테스트
=======================

# 공통 설정
- 대상: 시총 상위 300개 종목 (조건: 1년 데이터 존재)
- 리밸런싱: 매월 1일 (가격: 전월 종가)
- 벤치마크: 069500 (KODEX 200)
- 인버스 ETF: 114800 (KODEX 인버스)

# 지표 정의
- 13612MR: (1MR + 3MR + 6MR + 12MR) / 4  → 복합 모멘텀
- mean-R²: (1 + √RS3 + √RS6 + √RS12) / 4  → 추세 품질
- correlation marginal mean: 상관계수 행렬의 각 종목 평균값 → 분산 효과

# 전략 설명

Strategy 1 (Base)
  필터링: 13612MR 상위 1/2 | mean-R² 상위 1/2 | correlation 하위 1/3
  포지션: 1/N 동일 비중, 13612MR < 0 종목은 현금 보유
  종목 수: ~25개

Strategy 2 (Inverse)
  필터링: Strategy 1과 동일
  포지션: 1/N 동일 비중, 13612MR < 0 종목은 1/4 인버스 + 3/4 현금
  종목 수: ~25개

Strategy 3 ★ BEST
  필터링: 13612MR 상위 1/2 | mean-R² 상위 1/2 | correlation 하위 1/4
  포지션: 1/N 동일 비중, 13612MR < 0 종목은 현금 보유
  종목 수: ~19개
  특징: 분산 효과 강화 (correlation 1/4)

Strategy 4 ★ WORST
  필터링: 13612MR 상위 1/3 | mean-R² 상위 1/3 | correlation 하위 1/3
  포지션: 1/N 동일 비중, 13612MR < 0 종목은 현금 보유
  종목 수: ~11개
  특징: 모든 필터를 엄격하게 적용 (1/3)

Strategy 5
  필터링: 13612MR 상위 1/2 | mean-R² 상위 1/3 | correlation 하위 1/3
  포지션: 1/N 동일 비중, 13612MR < 0 종목은 현금 보유
  종목 수: ~16개
  특징: 추세 품질 필터 강화

Strategy 6
  필터링: 13612MR 상위 1/2 | mean-R² 상위 1/3 | correlation 하위 1/4
  포지션: 1/N 동일 비중, 13612MR < 0 종목은 현금 보유
  종목 수: ~12개
  특징: 추세 품질 + 분산 효과 강화

Strategy 7 (MACD Filter)
  필터링: 13612MR 상위 1/2 | mean-R² 상위 1/2 | correlation 하위 1/4
  포지션: 1/N 동일 비중, (13612MR < 0) or (MACD < 0) 종목은 현금 보유
  종목 수: ~19개 이하
  특징: Strategy 3 + MACD 오실레이터 필터 추가
         MACD = 3MR - 12MR (단기-장기 모멘텀 차이)
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


def select_portfolio_strategy5(momentum, correlation):
    """
    Strategy 5: 13612MR 상위 1/2 | mean-R2 상위 1/3 | correlation marginal mean 하위 1/3
    """
    momentum = momentum.dropna(subset=['13612MR', 'RS3', 'RS6', 'RS12'])
    if len(momentum) == 0:
        return {}

    total_stocks = len(momentum)

    # Step 1: 평균 모멘텀 상위 1/2
    avg_momentum = momentum['13612MR']
    step1_count = max(1, total_stocks // 2)
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


def select_portfolio_strategy6(momentum, correlation):
    """
    Strategy 6: 13612MR 상위 1/2 | mean-R2 상위 1/3 | correlation marginal mean 하위 1/4
    """
    momentum = momentum.dropna(subset=['13612MR', 'RS3', 'RS6', 'RS12'])
    if len(momentum) == 0:
        return {}

    total_stocks = len(momentum)

    # Step 1: 평균 모멘텀 상위 1/2
    avg_momentum = momentum['13612MR']
    step1_count = max(1, total_stocks // 2)
    step1_tickers = avg_momentum.nlargest(step1_count).index.tolist()

    # Step 2: 평균 R-squared 상위 1/3
    avg_rsquared = (1 + np.sqrt(momentum['RS3']) +
                    np.sqrt(momentum['RS6']) +
                    np.sqrt(momentum['RS12'])) / 4
    step2_candidates = avg_rsquared[step1_tickers]
    step2_count = max(1, len(step1_tickers) // 3)
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


def select_portfolio_strategy7(momentum, correlation):
    """
    Strategy 7: Strategy 3 변형 + MACD 필터
    - 13612MR 상위 1/2 | mean-R2 상위 1/2 | correlation marginal mean 하위 1/4
    - (13612MR < 0) or (MACD 오실레이터 < 0)인 종목은 현금 보유
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

    # 포트폴리오 구성 (MACD 오실레이터 조건 추가)
    n_stocks = len(step3_tickers)
    equal_weight = 1.0 / n_stocks if n_stocks > 0 else 0

    portfolio = {}
    for ticker in step3_tickers:
        # MACD 오실레이터: 3MR - 12MR (단기 모멘텀 - 장기 모멘텀)
        macd_osc = momentum.loc[ticker, '3MR'] - momentum.loc[ticker, '12MR']

        # (13612MR >= 0) and (MACD 오실레이터 >= 0) 인 경우에만 투자
        if momentum.loc[ticker, '13612MR'] >= 0 and macd_osc >= 0:
            portfolio[ticker] = equal_weight

    return portfolio


# ============================================================
# Main
# ============================================================

def main():
    print_step_header(0, "Strategy 1-7 백테스트")

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
    runner.add_strategy('strategy5', select_portfolio_strategy5)
    runner.add_strategy('strategy6', select_portfolio_strategy6)
    runner.add_strategy('strategy7', select_portfolio_strategy7)

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
