"""
KRX 모멘텀 전략 백테스트
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
  포지션: 1/N 동일 비중, (13612MR < 0) or (MACD Histogram < 0) 종목은 현금 보유
  종목 수: ~19개 이하
  특징: Strategy 3 + 표준 MACD 오실레이터 필터 추가
         MACD Histogram = (MACD Line - Signal Line)
         MACD Line = EMA(12) - EMA(26)
         Signal Line = EMA(9) of MACD Line
"""

import pandas as pd
import numpy as np
from core.file import import_dataframe_from_json
from core.config import settings
from core.utils import print_step_header, print_progress, print_completion, ensure_directory
from core.backtest_runner import BacktestRunner
from core.strategy import StrategyConfig, create_strategy_selector

SUBDIR = "backtest01"  # 현재 자기 자신 python file name


# ============================================================
# 전략 설정 (Strategy Configuration Pattern)
# ============================================================

STRATEGIES = [
    StrategyConfig(
        name="S223",
        momentum_ratio=1/2,
        rsquared_ratio=1/2,
        correlation_ratio=1/3,
        description="Base - 모멘텀 1/2 | R² 1/2 | 상관관계 1/3"
    ),
    StrategyConfig(
        name="S223MACD",
        momentum_ratio=1/2,
        rsquared_ratio=1/2,
        correlation_ratio=1/3,
        use_macd_filter=True,
        description="MACD Filter - Strategy 3 + MACD 오실레이터"
    ),
    # StrategyConfig(
    #     name="S01INV",
    #     momentum_ratio=1/2,
    #     rsquared_ratio=1/2,
    #     correlation_ratio=1/3,
    #     use_inverse=True,
    #     description="Inverse - Strategy 1 + 인버스 ETF"
    # ),
    StrategyConfig(
        name="S224",
        momentum_ratio=1/2,
        rsquared_ratio=1/2,
        correlation_ratio=1/4,
        description="추세 품질 + 분산 효과 강화"
    ),
    StrategyConfig(
        name="S224MACD",
        momentum_ratio=1/2,
        rsquared_ratio=1/2,
        correlation_ratio=1/4,
        use_macd_filter=True,
        description="MACD Filter - Strategy 3 + MACD 오실레이터"
    ),
    StrategyConfig(
        name="S233",
        momentum_ratio=1/2,
        rsquared_ratio=1/3,
        correlation_ratio=1/3,
        description="추세 품질 + 분산 효과 강화"
    ),
    StrategyConfig(
        name="S233MACD",
        momentum_ratio=1/2,
        rsquared_ratio=1/3,
        correlation_ratio=1/3,
        use_macd_filter=True,
        description="MACD Filter - Strategy 3 + MACD 오실레이터"
    ),
    StrategyConfig(
        name="S234",
        momentum_ratio=1/2,
        rsquared_ratio=1/3,
        correlation_ratio=1/4,
        description="추세 품질 + 분산 효과 강화"
    ),
    StrategyConfig(
        name="S234MACD",
        momentum_ratio=1/2,
        rsquared_ratio=1/3,
        correlation_ratio=1/4,
        use_macd_filter=True,
        description="MACD Filter - Strategy 3 + MACD 오실레이터"
    ),
    StrategyConfig(
        name="S432",
        momentum_ratio=1/4,
        rsquared_ratio=1/3,
        correlation_ratio=1/2,
        description="추세 품질 + 분산 효과 강화"
    ),
    StrategyConfig(
        name="S432MACD",
        momentum_ratio=1/4,
        rsquared_ratio=1/3,
        correlation_ratio=1/2,
        use_macd_filter=True,
        description="MACD Filter - Strategy 3 + MACD 오실레이터"
    ),
]


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

    for config in STRATEGIES:
        # Factory 패턴으로 전략 선택 함수 생성
        selector = create_strategy_selector(config)

        # 백테스트 실행 (MACD 필요 여부 전달)
        runner.add_strategy(config.name, selector, config.use_inverse, config.use_macd_filter)

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
