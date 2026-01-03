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

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from core.file import import_dataframe_from_json, export_with_message
from core.finance import get_corrMatrix
from core.models import LM
from core.config import settings
from core.utils import print_step_header, print_progress, print_completion, ensure_directory


# ============================================================
# Signal 계산 함수
# ============================================================

def calculate_signals_at_date(closeM_log, closeM, end_idx):
    """
    특정 시점까지의 데이터로 momentum과 correlation 계산

    Parameters:
    -----------
    closeM_log : pd.DataFrame
        로그 가격 데이터
    closeM : pd.DataFrame
        가격 데이터
    end_idx : int
        계산할 마지막 인덱스

    Returns:
    --------
    tuple
        (momentum DataFrame, correlation DataFrame)
    """
    # 해당 시점까지의 데이터만 사용
    prices_log = closeM_log.iloc[:end_idx+1]
    prices = closeM.iloc[:end_idx+1]

    # Momentum 계산
    momentum = pd.DataFrame(index=prices.columns)

    # 1~12개월 수익률
    for i in range(1, 13):
        if len(prices) >= i + 1:
            returns = prices.pct_change(periods=i).iloc[-1]
            momentum[f'{i}MR'] = returns
        else:
            momentum[f'{i}MR'] = np.nan

    # 평균 모멘텀
    momentum['13612MR'] = (momentum['1MR'] + momentum['3MR'] +
                           momentum['6MR'] + momentum['12MR']) / 4

    # Linear Regression (RS3, RS6, RS12만 계산)
    for period in [3, 6, 12]:
        if len(prices_log) >= period:
            LR = LM().fit(prices_log, period)
            momentum[f'AS{period}'] = (np.exp(LR.slope * 12) - 1)
            momentum[f'RS{period}'] = LR.score
        else:
            momentum[f'AS{period}'] = np.nan
            momentum[f'RS{period}'] = np.nan

    # Correlation 계산 (최근 12개월)
    corr_periods = settings.signals.correlation.periods
    if len(prices) >= corr_periods:
        correlation = get_corrMatrix(prices, corr_periods)
    else:
        # 데이터 부족 시 빈 상관관계 행렬
        correlation = pd.DataFrame(index=prices.columns, columns=prices.columns)
        correlation[:] = np.nan

    return momentum, correlation


# ============================================================
# Strategy 1 포트폴리오 선택
# ============================================================

def select_portfolio_strategy1(momentum, correlation):
    """
    Strategy 1에 따른 포트폴리오 선택

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 지표
    correlation : pd.DataFrame
        Correlation matrix

    Returns:
    --------
    dict
        {ticker: weight} 형태의 포트폴리오
    """
    # NaN 제거
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

    # 선택된 종목들만 추출
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
        # 음수 모멘텀은 0 비중 (현금)

    return portfolio


# ============================================================
# 성과 지표 계산
# ============================================================

def calculate_metrics(returns, benchmark_returns=None):
    """
    성과 지표 계산

    Parameters:
    -----------
    returns : pd.Series
        월별 수익률
    benchmark_returns : pd.Series, optional
        벤치마크 수익률

    Returns:
    --------
    dict
        성과 지표들
    """
    # 누적 수익률
    cumulative_returns = (1 + returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1

    # CAGR (Compound Annual Growth Rate)
    n_years = len(returns) / 12
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # MDD (Maximum Drawdown)
    cummax = cumulative_returns.cummax()
    drawdown = (cumulative_returns - cummax) / cummax
    mdd = drawdown.min()

    # Volatility (연율화)
    volatility = returns.std() * np.sqrt(12)

    # Sharpe Ratio (무위험 수익률 0 가정)
    sharpe = cagr / volatility if volatility > 0 else 0

    # Sortino Ratio (하방 편차만 고려)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(12) if len(downside_returns) > 0 else 0
    sortino = cagr / downside_std if downside_std > 0 else 0

    # Win Rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    metrics = {
        'Total Return': total_return,
        'CAGR': cagr,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'MDD': mdd,
        'Win Rate': win_rate,
        'N Periods': len(returns)
    }

    # 벤치마크 대비 지표
    if benchmark_returns is not None:
        # Tracking Error
        excess_returns = returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(12)
        metrics['Tracking Error'] = tracking_error

        # Information Ratio
        excess_return = excess_returns.mean() * 12
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
        metrics['Information Ratio'] = information_ratio

        # Alpha (초과 수익)
        benchmark_cagr = (1 + (1 + benchmark_returns).prod() - 1) ** (1 / n_years) - 1 if n_years > 0 else 0
        metrics['Alpha'] = cagr - benchmark_cagr

    return metrics


# ============================================================
# 백테스트 실행
# ============================================================

def run_backtest(closeM, end_date=None):
    """
    Strategy 1 백테스트 실행

    Parameters:
    -----------
    closeM : pd.DataFrame
        월별 종가 데이터
    end_date : str or datetime, optional
        백테스트 종료일 (YYYY-MM-DD 형식, None이면 데이터 끝까지)

    Returns:
    --------
    tuple
        (portfolio_values, benchmark_values, monthly_returns, holdings_history)
    """
    closeM_log = np.log(closeM)

    # 최소 12개월 이후부터 백테스트 시작
    start_idx = 12

    # 종료 인덱스 결정
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        # end_date 이하인 마지막 인덱스 찾기
        valid_dates = closeM.index[closeM.index <= end_date]
        if len(valid_dates) > 0:
            end_idx = closeM.index.get_loc(valid_dates[-1])
        else:
            print(f"  경고: 종료일 {end_date}이 데이터 범위 밖입니다. 전체 기간 사용.")
            end_idx = len(closeM) - 1
    else:
        end_idx = len(closeM) - 1

    # 백테스트 결과 저장
    portfolio_values = []
    monthly_returns = []
    holdings_history = []
    dates = []

    current_value = 1.0  # 초기 포트폴리오 가치

    print(f"\n백테스트 기간: {closeM.index[start_idx]} ~ {closeM.index[end_idx]}")
    print(f"총 {end_idx - start_idx}개월\n")

    for i in range(start_idx, end_idx):
        # t월 종가로 signal 계산
        signal_date = closeM.index[i]
        momentum, correlation = calculate_signals_at_date(closeM_log, closeM, i)

        # 포트폴리오 선택
        portfolio = select_portfolio_strategy1(momentum, correlation)

        # t+1월 수익률 계산
        next_date = closeM.index[i + 1]
        next_returns = closeM.iloc[i + 1] / closeM.iloc[i] - 1

        # 포트폴리오 수익률
        if len(portfolio) > 0:
            portfolio_return = sum(weight * next_returns[ticker]
                                 for ticker, weight in portfolio.items())
        else:
            portfolio_return = 0.0  # 전액 현금

        # 포트폴리오 가치 업데이트
        current_value *= (1 + portfolio_return)

        # 기록
        dates.append(next_date)
        portfolio_values.append(current_value)
        monthly_returns.append(portfolio_return)
        holdings_history.append({
            'date': next_date,
            'holdings': portfolio.copy(),
            'n_stocks': len(portfolio),
            'cash_weight': 1.0 - sum(portfolio.values())
        })

        # 진행 상황 출력
        if i == start_idx or (i - start_idx + 1) % 12 == 0:
            print(f"  {signal_date.strftime('%Y-%m')}: 포트폴리오 가치 = {current_value:.4f} "
                  f"({len(portfolio)}개 종목)")

    return (pd.Series(portfolio_values, index=dates),
            pd.Series(monthly_returns, index=dates),
            holdings_history)


def get_krx300_benchmark(start_date, end_date):
    """
    KRX300 지수 데이터 가져오기

    Parameters:
    -----------
    start_date : datetime
        시작일
    end_date : datetime
        종료일

    Returns:
    --------
    pd.Series
        KRX300 월말 종가
    """
    try:
        # KRX300 지수 다운로드
        krx300 = fdr.DataReader('KRX300', start_date, end_date)
        krx300.index = pd.to_datetime(krx300.index)

        # 월말 종가만 추출
        krx300_monthly = krx300['Close'].resample('ME').last()

        return krx300_monthly
    except Exception as e:
        print(f"  경고: KRX300 데이터를 가져올 수 없습니다: {e}")
        return None


# ============================================================
# Main
# ============================================================

def main():
    print_step_header(0, "Strategy 1 백테스트")

    # 설정 로드
    price_dir = settings.output.price_dir.path
    backtest_base_dir = settings.output.backtest_dir.path
    strategy_name = "strategy1"
    output_dir = f"{backtest_base_dir}/{strategy_name}"
    end_date = settings.backtest.end_date if hasattr(settings, 'backtest') else None
    ensure_directory(output_dir)

    # 1. 데이터 로드
    print_progress(1, 5, "가격 데이터 로드...")
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

    # 2. 백테스트 실행
    print_progress(2, 5, "백테스트 실행 중...")
    portfolio_values, monthly_returns, holdings_history = run_backtest(closeM, end_date)

    # 3. 벤치마크 데이터 로드
    print_progress(3, 5, "벤치마크(KRX300) 데이터 로드...")
    benchmark_prices = get_krx300_benchmark(portfolio_values.index[0], portfolio_values.index[-1])

    if benchmark_prices is not None:
        # 벤치마크 수익률 계산
        benchmark_returns = benchmark_prices.pct_change().dropna()
        # 날짜 정렬
        benchmark_returns = benchmark_returns.reindex(monthly_returns.index)

        # 벤치마크 정규화 (시작 = 1.0)
        benchmark_values = (1 + benchmark_returns.fillna(0)).cumprod()
        print(f"      벤치마크 로드 완료")
    else:
        benchmark_returns = None
        benchmark_values = None

    # 4. 성과 지표 계산
    print_progress(4, 5, "성과 지표 계산...")
    metrics = calculate_metrics(monthly_returns, benchmark_returns)

    # 결과 출력
    print("\n" + "="*70)
    print("백테스트 결과")
    print("="*70)
    print(f"최종 포트폴리오 가치: {portfolio_values.iloc[-1]:.4f}")
    print(f"총 수익률: {metrics['Total Return']:.2%}")
    print(f"CAGR: {metrics['CAGR']:.2%}")
    print(f"변동성(연율화): {metrics['Volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")
    print(f"Sortino Ratio: {metrics['Sortino Ratio']:.4f}")
    print(f"MDD: {metrics['MDD']:.2%}")
    print(f"Win Rate: {metrics['Win Rate']:.2%}")

    if benchmark_returns is not None:
        print(f"\n벤치마크 대비:")
        print(f"Alpha: {metrics['Alpha']:.2%}")
        print(f"Tracking Error: {metrics['Tracking Error']:.2%}")
        print(f"Information Ratio: {metrics['Information Ratio']:.4f}")

    # 5. 결과 저장
    print_progress(5, 5, "결과 저장...")

    # 포트폴리오 가치 시계열
    results_df = pd.DataFrame({
        'Portfolio': portfolio_values,
        'Returns': monthly_returns
    })
    if benchmark_values is not None:
        results_df['Benchmark'] = benchmark_values
        results_df['Benchmark_Returns'] = benchmark_returns

    export_with_message(results_df, f'{output_dir}/results', 'Backtest Results')

    # 성과 지표
    metrics_df = pd.DataFrame([metrics]).T
    metrics_df.columns = ['Value']
    export_with_message(metrics_df, f'{output_dir}/metrics', 'Performance Metrics')

    # 보유 종목 히스토리 (샘플: 매 12개월마다)
    holdings_sample = [h for i, h in enumerate(holdings_history) if i % 12 == 0]
    holdings_df = pd.DataFrame([
        {
            'Date': h['date'],
            'N_Stocks': h['n_stocks'],
            'Cash_Weight': h['cash_weight'],
            'Holdings': ', '.join([f"{t}({w:.2%})" for t, w in sorted(h['holdings'].items(), key=lambda x: -x[1])[:10]])
        }
        for h in holdings_sample
    ])
    export_with_message(holdings_df, f'{output_dir}/holdings', 'Holdings History (Sample)')

    print_completion(0)


if __name__ == "__main__":
    main()
