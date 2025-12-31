"""
STEP 4: 종목 선택 및 포트폴리오 구성
- Momentum/Performance 지표 기반 종목 필터링 및 순위 매기기
- 선택된 종목으로 포트폴리오 구성
"""

import pandas as pd
import numpy as np
from core.file import import_dataframe_from_json, export_dataframe_to_formats
from core.config import settings


def calculate_composite_scores(momentum, performance):
    """
    복합 점수 계산

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 데이터
    performance : pd.DataFrame
        Performance 데이터

    Returns:
    --------
    pd.DataFrame
        복합 점수 데이터
    """
    scores = pd.DataFrame(index=momentum.index)

    # TODO: 복합 점수 계산 로직 구현
    # 예시:
    # - Momentum Quality (R × AS)
    # - Sharpe Ratio
    # - Sortino Ratio
    # - 가중 평균 점수 등

    return scores


def select_stocks(scores, criteria):
    """
    종목 선택 및 순위 매기기

    Parameters:
    -----------
    scores : pd.DataFrame
        복합 점수 데이터
    criteria : dict
        선택 기준 (임계값, 개수 등)

    Returns:
    --------
    pd.DataFrame
        선택된 종목 리스트 (순위 포함)
    """
    selected = pd.DataFrame(index=scores.index)

    # TODO: 종목 선택 로직 구현
    # 예시:
    # - 상위 N개 종목 선택
    # - 특정 점수 이상 종목 선택
    # - 다중 조건 필터링

    return selected


def construct_portfolio(selected_stocks, weights_method='equal'):
    """
    포트폴리오 구성

    Parameters:
    -----------
    selected_stocks : pd.DataFrame
        선택된 종목 리스트
    weights_method : str
        가중치 계산 방법 ('equal', 'score_weighted', 'risk_parity' 등)

    Returns:
    --------
    pd.DataFrame
        포트폴리오 구성 (종목, 가중치)
    """
    portfolio = pd.DataFrame(index=selected_stocks.index)

    # TODO: 포트폴리오 가중치 계산 로직 구현
    # 예시:
    # - Equal weight: 1/N
    # - Score weighted: 점수 비율
    # - Risk parity: 리스크 기반 가중치

    if weights_method == 'equal':
        n_stocks = len(selected_stocks)
        portfolio['weight'] = 1.0 / n_stocks if n_stocks > 0 else 0

    return portfolio


def main():
    print("=" * 70)
    print("STEP 4: 종목 선택 및 포트폴리오 구성")
    print("=" * 70)

    # 설정 로드
    signal_dir = settings.output.signal_dir
    # output_dir = settings.output.portfolio_dir  # TODO: settings.yaml에 추가 필요

    # 1. Signal 데이터 로드
    print("\n[1/4] Signal 데이터 로드...")
    momentum = import_dataframe_from_json(f'{signal_dir}/momentum.json')
    performance = import_dataframe_from_json(f'{signal_dir}/performance.json')
    print(f"      Momentum: {momentum.shape}")
    print(f"      Performance: {performance.shape}")

    # 2. 복합 점수 계산
    print("\n[2/4] 복합 점수 계산...")
    scores = calculate_composite_scores(momentum, performance)
    print(f"      완료: {scores.shape}")

    # 3. 종목 선택
    print("\n[3/4] 종목 선택...")
    criteria = {}  # TODO: 선택 기준 정의
    selected = select_stocks(scores, criteria)
    print(f"      선택된 종목: {len(selected)}개")

    # 4. 포트폴리오 구성
    print("\n[4/4] 포트폴리오 구성...")
    portfolio = construct_portfolio(selected, weights_method='equal')
    print(f"      완료: {portfolio.shape}")

    # 5. 저장
    print("\n파일 저장 (HTML, TSV, JSON)...")

    print("  scores:")
    export_dataframe_to_formats(scores, f'{signal_dir}/scores', 'Composite Scores')

    print("  selected:")
    export_dataframe_to_formats(selected, f'{signal_dir}/selected', 'Selected Stocks')

    print("  portfolio:")
    export_dataframe_to_formats(portfolio, f'{signal_dir}/portfolio', 'Portfolio Composition')

    print("\n" + "=" * 70)
    print("STEP 4 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
