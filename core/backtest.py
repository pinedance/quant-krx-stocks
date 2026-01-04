"""
백테스트 공통 함수 및 클래스
- 시그널 계산, 성과 지표, 백테스트 실행 등 재사용 가능한 로직
"""

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from typing import Callable, Optional, Dict, Tuple, List
from datetime import datetime
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from core.finance import get_corr_matrix
from core.models import LM
from core.config import settings


# ============================================================
# Signal 계산 (전체 시계열)
# ============================================================

def calculate_all_momentum(closeM: pd.DataFrame, closeM_log: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    전체 시계열의 모멘텀 지표 계산 (lookahead bias 방지)
    pandas 벡터 연산 사용으로 고속 처리

    Parameters:
    -----------
    closeM : pd.DataFrame
        가격 데이터 (rows=dates, cols=tickers)
    closeM_log : pd.DataFrame
        로그 가격 데이터
    verbose : bool
        진행 상황 출력 여부

    Returns:
    --------
    tuple
        (mmt_13612MR, rs_3, rs_6, rs_12) - 각각 (rows=dates, cols=tickers)
    """
    if verbose:
        print(f"      모멘텀 지표 계산 중... (벡터 연산)")

    # 1. 13612MR 계산 (벡터 연산, lookahead bias 자동 방지)
    mr_1 = closeM.pct_change(periods=1)
    mr_3 = closeM.pct_change(periods=3)
    mr_6 = closeM.pct_change(periods=6)
    mr_12 = closeM.pct_change(periods=12)

    mmt_13612MR = (mr_1 + mr_3 + mr_6 + mr_12) / 4

    if verbose:
        print(f"        ✓ 13612MR 계산 완료")

    # 2. mean-R² 계산: (1 + √RS3 + √RS6 + √RS12) / 4
    # Rolling window로 각 period별 R² 계산 (최적화: 벡터 연산)
    rs_3 = pd.DataFrame(index=closeM_log.index, columns=closeM_log.columns, dtype=float)
    rs_6 = pd.DataFrame(index=closeM_log.index, columns=closeM_log.columns, dtype=float)
    rs_12 = pd.DataFrame(index=closeM_log.index, columns=closeM_log.columns, dtype=float)

    if verbose:
        print(f"        R² 계산 중... (numpy 벡터 연산)")

    # 각 period별로 rolling R² 계산 (numpy 벡터화)
    for period, rs_result in [(3, rs_3), (6, rs_6), (12, rs_12)]:
        if len(closeM_log) < period:
            continue

        # X는 모든 window에 공통
        x = np.arange(period)
        x_mean = (period - 1) / 2
        x_centered = x - x_mean
        x_var = np.sum(x_centered ** 2)

        # 각 종목별로 계산 (종목별 loop는 유지, 시점별은 벡터화)
        for ticker in closeM_log.columns:
            prices = closeM_log[ticker]  # Series로 유지 (index 보존)
            n = len(prices)

            if n < period:
                continue

            # Rolling window 계산
            for i in range(period - 1, n):
                y = prices.iloc[i - period + 1:i + 1].values  # 해당 window의 값만

                # NaN 체크
                if not np.all(np.isfinite(y)):
                    continue

                # Linear regression (vectorized)
                y_mean = np.mean(y)
                y_centered = y - y_mean

                # Slope
                numerator = np.sum(x_centered * y_centered)
                slope = numerator / x_var

                # R²
                y_pred = slope * x + (y_mean - slope * x_mean)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum(y_centered ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0

                # i는 prices의 iloc 인덱스, 날짜 인덱스로 저장
                date_idx = prices.index[i]
                rs_result.loc[date_idx, ticker] = r2

    if verbose:
        print(f"        ✓ R² 계산 완료")

    if verbose:
        print(f"      모멘텀 지표 계산 완료!")

    return mmt_13612MR, rs_3, rs_6, rs_12


def calculate_all_macd(closeM: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, verbose: bool = True) -> pd.DataFrame:
    """
    전체 시계열의 MACD Histogram 계산 (lookahead bias 방지)
    DataFrame 전체 벡터 연산으로 고속 처리

    Parameters:
    -----------
    closeM : pd.DataFrame
        가격 데이터 (rows=dates, cols=tickers)
    fast_period : int
        단기 EMA 기간
    slow_period : int
        장기 EMA 기간
    signal_period : int
        Signal Line EMA 기간
    verbose : bool
        진행 상황 출력 여부

    Returns:
    --------
    pd.DataFrame
        MACD Histogram (rows=dates, cols=tickers)
    """
    if verbose:
        print(f"      MACD Histogram 계산 중... (전체 DataFrame 벡터 연산)")

    # EMA 계산 (전체 DataFrame에 대해 한번에, ewm은 자동으로 lookahead bias 방지)
    ema_fast = closeM.ewm(span=fast_period, adjust=False).mean()
    ema_slow = closeM.ewm(span=slow_period, adjust=False).mean()

    # MACD Line = EMA(fast) - EMA(slow)
    macd_line = ema_fast - ema_slow

    # Signal Line = EMA(signal_period) of MACD Line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # MACD Histogram = MACD Line - Signal Line
    macd_histogram = macd_line - signal_line

    if verbose:
        print(f"      MACD Histogram 계산 완료!")

    return macd_histogram


# ============================================================
# Signal 계산 (단일 시점 - 하위 호환성 유지)
# ============================================================

def calculate_signals_at_date(closeM_log: pd.DataFrame, closeM: pd.DataFrame, end_idx: int, include_macd: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    include_macd : bool
        MACD Histogram 계산 여부 (기본값: True)

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

    # MACD Histogram 계산 (필요할 때만)
    if include_macd:
        if len(prices) >= 26:  # slow_period
            macd_hist = calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9)
            momentum['MACD_Histogram'] = macd_hist.iloc[-1]
        else:
            momentum['MACD_Histogram'] = np.nan
    else:
        # MACD 계산 안 함 (성능 최적화)
        momentum['MACD_Histogram'] = np.nan

    # Correlation 계산 (최근 12개월)
    corr_periods = settings.signals.correlation.periods
    if len(prices) >= corr_periods:
        correlation = get_corr_matrix(prices, corr_periods)
    else:
        # 데이터 부족 시 빈 상관관계 행렬
        correlation = pd.DataFrame(index=prices.columns, columns=prices.columns)
        correlation[:] = np.nan

    return momentum, correlation


def calculate_macd(closeM: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """
    표준 MACD 지표 계산 (MACD Line, Signal Line, MACD Histogram)

    Parameters:
    -----------
    closeM : pd.DataFrame
        월별 종가 데이터 (행: 날짜, 열: 종목)
    fast_period : int
        단기 EMA 기간 (기본값: 12개월)
    slow_period : int
        장기 EMA 기간 (기본값: 26개월)
    signal_period : int
        Signal Line EMA 기간 (기본값: 9개월)

    Returns:
    --------
    pd.DataFrame
        MACD Histogram 값 (행: 날짜, 열: 종목)
        양수: 상승 모멘텀, 음수: 하락 모멘텀
    """
    # 각 종목별로 MACD 계산
    macd_histogram = pd.DataFrame(index=closeM.index, columns=closeM.columns)

    for ticker in closeM.columns:
        prices = closeM[ticker].dropna()

        if len(prices) < slow_period:
            # 데이터가 충분하지 않으면 NaN
            continue

        # EMA 계산
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

        # MACD Line = EMA(12) - EMA(26)
        macd_line = ema_fast - ema_slow

        # Signal Line = EMA(9) of MACD Line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # MACD Histogram = MACD Line - Signal Line
        macd_hist = macd_line - signal_line

        # 결과 저장
        macd_histogram.loc[prices.index, ticker] = macd_hist

    return macd_histogram


# ============================================================
# 성과 지표 계산
# ============================================================

def calculate_metrics(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
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
# ETF 데이터 로드
# ============================================================

def get_etf_data(ticker: str, start_date, end_date) -> Optional[pd.Series]:
    """
    ETF 월말 종가 데이터 가져오기

    Parameters:
    -----------
    ticker : str
        ETF 티커 (예: '069500', '114800')
    start_date : datetime
        시작일
    end_date : datetime
        종료일

    Returns:
    --------
    pd.Series or None
        ETF 월말 종가
    """
    try:
        # ETF 데이터 다운로드
        etf_data = fdr.DataReader(ticker, start_date, end_date)
        etf_data.index = pd.to_datetime(etf_data.index)

        # 월말 종가만 추출
        etf_monthly = etf_data['Close'].resample('ME').last()

        return etf_monthly
    except Exception as e:
        print(f"  경고: {ticker} 데이터를 가져올 수 없습니다: {e}")
        return None


# ============================================================
# 백테스트 실행
# ============================================================

def run_backtest(
    closeM: pd.DataFrame,
    strategy_selector: Callable,
    inverse_etf_prices: Optional[pd.Series] = None,
    end_date: Optional[str] = None,
    verbose: bool = True,
    signal_provider: Optional[Callable] = None
) -> Tuple[pd.Series, pd.Series, List[Dict]]:
    """
    백테스트 실행 (일반 전략 및 인버스 전략 지원)

    Parameters:
    -----------
    closeM : pd.DataFrame
        월별 종가 데이터
    strategy_selector : function
        포트폴리오 선택 함수 (momentum, correlation을 받아서 portfolio 반환)
    inverse_etf_prices : pd.Series, optional
        인버스 ETF 가격 (포트폴리오에 'INVERSE' 키가 있으면 사용)
    end_date : str or datetime, optional
        백테스트 종료일
    verbose : bool
        진행 상황 출력 여부
    signal_provider : Callable, optional
        시그널 제공 함수 (end_idx를 받아서 momentum, correlation 반환)
        None이면 기본 calculate_signals_at_date 사용

    Returns:
    --------
    tuple
        (portfolio_values, monthly_returns, holdings_history)
    """
    closeM_log = np.log(closeM)

    # Signal provider 설정 (없으면 기본 함수 사용)
    if signal_provider is None:
        signal_provider = lambda end_idx: calculate_signals_at_date(closeM_log, closeM, end_idx, include_macd=True)

    # 최소 12개월 이후부터 백테스트 시작
    start_idx = 12 + 1

    # 종료 인덱스 결정
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        valid_dates = closeM.index[closeM.index <= end_date]
        if len(valid_dates) > 0:
            end_idx = closeM.index.get_loc(valid_dates[-1])
        else:
            if verbose:
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

    if verbose:
        print(f"\n백테스트 기간: {closeM.index[start_idx]} ~ {closeM.index[end_idx]}")
        print(f"총 {end_idx - start_idx + 1}개월\n")

    # 시작일(t=0)에 초기값 추가
    start_date = closeM.index[start_idx]
    dates.append(start_date)
    portfolio_values.append(1.0)
    monthly_returns.append(0.0)

    for i in range(start_idx, end_idx):
        # t월 종가로 signal 계산
        signal_date = closeM.index[i]
        momentum, correlation = signal_provider(i)

        # 포트폴리오 선택
        portfolio = strategy_selector(momentum, correlation)

        # t+1월 수익률 계산
        next_date = closeM.index[i + 1]
        next_returns = closeM.iloc[i + 1] / closeM.iloc[i] - 1

        # 포트폴리오 수익률 계산
        portfolio_return = 0.0
        if len(portfolio) > 0:
            for ticker, weight in portfolio.items():
                if ticker == 'INVERSE' and inverse_etf_prices is not None:
                    # 인버스 ETF 수익률
                    if next_date in inverse_etf_prices.index and signal_date in inverse_etf_prices.index:
                        inverse_return = inverse_etf_prices.loc[next_date] / inverse_etf_prices.loc[signal_date] - 1
                        portfolio_return += weight * inverse_return
                elif ticker != 'INVERSE':
                    # 일반 주식
                    portfolio_return += weight * next_returns[ticker]

        # 포트폴리오 가치 업데이트
        current_value *= (1 + portfolio_return)

        # 기록
        dates.append(next_date)
        portfolio_values.append(current_value)
        monthly_returns.append(portfolio_return)

        # Holdings 기록
        inverse_weight = portfolio.get('INVERSE', 0.0)
        stock_portfolio = {k: v for k, v in portfolio.items() if k != 'INVERSE'}

        holdings_history.append({
            'date': next_date,
            'holdings': stock_portfolio.copy(),
            'n_stocks': len(stock_portfolio),
            'inverse_weight': inverse_weight,
            'cash_weight': 1.0 - sum(portfolio.values())
        })

        # 진행 상황 출력
        if verbose and (i == start_idx or (i - start_idx + 1) % 12 == 0):
            if inverse_weight > 0:
                print(f"  {signal_date.strftime('%Y-%m')}: 포트폴리오 가치 = {current_value:.4f} "
                      f"({len(stock_portfolio)}개 종목, 인버스 {inverse_weight:.2%})")
            else:
                print(f"  {signal_date.strftime('%Y-%m')}: 포트폴리오 가치 = {current_value:.4f} "
                      f"({len(stock_portfolio)}개 종목)")

    return (pd.Series(portfolio_values, index=dates),
            pd.Series(monthly_returns, index=dates),
            holdings_history)


# ============================================================
# 결과 저장
# ============================================================

def save_backtest_results(
    output_dir: str,
    strategies: Dict[str, Dict],
    benchmark_returns: Optional[pd.Series] = None
) -> None:
    """
    백테스트 결과를 파일로 저장

    Parameters:
    -----------
    output_dir : str
        저장 디렉토리
    strategies : dict
        {'strategy_name': {'returns': Series, 'values': Series, 'metrics': dict}}
    benchmark_returns : pd.Series, optional
        벤치마크 수익률 (전략과 동일한 기간으로 필터링된 것)
    """
    # metrics.tsv 생성
    metric_names = ['Total Return', 'CAGR', 'Volatility', 'Sharpe Ratio',
                    'Sortino Ratio', 'MDD', 'Win Rate', 'N Periods']

    metrics_data = {}
    for name, data in strategies.items():
        metrics_data[name] = [data['metrics'].get(m, np.nan) for m in metric_names]

    # 벤치마크 추가 (이미 필터링된 benchmark_returns를 받음)
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        benchmark_metrics = calculate_metrics(benchmark_returns)
        metrics_data['benchmark'] = [benchmark_metrics.get(m, np.nan) for m in metric_names]

    # 컬럼 순서: benchmark, strategy1, strategy2, ...
    if benchmark_returns is not None:
        column_order = ['benchmark'] + [k for k in strategies.keys()]
        metrics_df = pd.DataFrame(metrics_data, index=metric_names)[column_order]
    else:
        metrics_df = pd.DataFrame(metrics_data, index=metric_names)

    # TSV 파일로 저장
    metrics_path = f'{output_dir}/metrics.tsv'
    metrics_df.to_csv(metrics_path, sep='\t')
    print(f"      metrics 저장 완료: {metrics_path}")

    # 월별 수익률 비교 데이터 저장
    returns_comparison = {}
    if benchmark_returns is not None:
        first_strategy = list(strategies.values())[0]
        returns_comparison['benchmark'] = benchmark_returns.reindex(
            first_strategy['returns'].index
        ).fillna(0)

    for name, data in strategies.items():
        returns_comparison[name] = data['returns']

    # Portfolio values 추가
    for name, data in strategies.items():
        returns_comparison[f'portfolio_value_{name}'] = data['values']

    returns_df = pd.DataFrame(returns_comparison)
    returns_path = f'{output_dir}/monthly_returns.tsv'
    returns_df.to_csv(returns_path, sep='\t')
    print(f"      monthly returns 저장 완료: {returns_path}")


# ============================================================
# 백테스트 러너 클래스
# ============================================================

class BacktestRunner:
    """
    백테스트 실행을 관리하는 클래스
    """

    def __init__(self, closeM: pd.DataFrame, output_dir: str, end_date: Optional[str] = None):
        """
        Parameters:
        -----------
        closeM : pd.DataFrame
            월별 종가 데이터
        output_dir : str
            결과 저장 디렉토리
        end_date : str, optional
            백테스트 종료일
        """
        self.closeM = closeM
        self.closeM_log = np.log(closeM)
        self.output_dir = output_dir
        self.end_date = end_date
        self.strategies = {}
        self.benchmark_prices = None
        self.benchmark_returns = None
        self.inverse_etf_prices = None
        self._correlation_cache = {}  # Correlation 캐시: end_idx -> correlation_matrix

        # Pre-compute: 전체 시계열 지표 계산
        print("\n" + "="*70)
        print("지표 사전 계산 (Pre-computation)")
        print("="*70)

        # 1. 모멘텀 지표 계산
        self.mmt_13612MR, self.rs_3, self.rs_6, self.rs_12 = calculate_all_momentum(
            self.closeM,
            self.closeM_log,
            verbose=True
        )

        # 2. MACD Histogram 계산 (필요 시에만 - 나중에 lazy compute 가능)
        self.macd_histogram = None  # 필요할 때 계산

        print("="*70)

    def load_benchmark(self, ticker: str = '069500') -> None:
        """벤치마크 ETF 데이터 로드"""
        self.benchmark_prices = get_etf_data(ticker, self.closeM.index[0], self.closeM.index[-1])
        if self.benchmark_prices is not None:
            self.benchmark_returns = self.benchmark_prices.pct_change().dropna()
            print(f"      벤치마크({ticker}) 로드 완료")
        else:
            print(f"      경고: 벤치마크 데이터 없음")

    def load_inverse_etf(self, ticker: str = '114800') -> None:
        """인버스 ETF 데이터 로드"""
        self.inverse_etf_prices = get_etf_data(ticker, self.closeM.index[0], self.closeM.index[-1])
        if self.inverse_etf_prices is not None:
            print(f"      인버스 ETF({ticker}) 로드 완료")
        else:
            print(f"      경고: 인버스 ETF 데이터 없음")

    def _ensure_macd_computed(self) -> None:
        """MACD가 필요할 때 lazy하게 계산"""
        if self.macd_histogram is None:
            print("\n" + "="*70)
            print("MACD Histogram 계산 (첫 사용 시)")
            print("="*70)
            self.macd_histogram = calculate_all_macd(
                self.closeM,
                fast_period=12,
                slow_period=26,
                signal_period=9,
                verbose=True
            )
            print("="*70)

    def _get_correlation_at_date(self, end_idx: int) -> pd.DataFrame:
        """특정 시점의 correlation matrix 가져오기 (캐시 사용)"""
        if end_idx in self._correlation_cache:
            return self._correlation_cache[end_idx]

        # 캐시에 없으면 계산
        corr_periods = settings.signals.correlation.periods
        prices = self.closeM.iloc[:end_idx+1]

        if len(prices) >= corr_periods:
            correlation = get_corr_matrix(prices, corr_periods)
        else:
            # 데이터 부족 시 빈 상관관계 행렬
            correlation = pd.DataFrame(index=prices.columns, columns=prices.columns)
            correlation[:] = np.nan

        # 캐시에 저장
        self._correlation_cache[end_idx] = correlation

        return correlation

    def _create_signal_provider(self, needs_macd: bool) -> Callable:
        """
        Pre-computed 데이터를 사용하는 시그널 제공자 생성

        Parameters:
        -----------
        needs_macd : bool
            MACD 계산 필요 여부

        Returns:
        --------
        Callable
            end_idx를 받아 (momentum, correlation)을 반환하는 함수
        """
        # MACD가 필요하면 미리 계산
        if needs_macd:
            self._ensure_macd_computed()

        def signal_provider(end_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
            # Pre-computed 데이터에서 해당 시점 slice
            date = self.closeM.index[end_idx]
            tickers = self.closeM.columns

            # Momentum DataFrame 구성 (rows=tickers, cols=indicators)
            momentum = pd.DataFrame(index=tickers)
            momentum['13612MR'] = self.mmt_13612MR.loc[date]
            momentum['RS3'] = self.rs_3.loc[date]
            momentum['RS6'] = self.rs_6.loc[date]
            momentum['RS12'] = self.rs_12.loc[date]

            if needs_macd and self.macd_histogram is not None:
                momentum['MACD_Histogram'] = self.macd_histogram.loc[date]
            else:
                momentum['MACD_Histogram'] = np.nan

            # Correlation matrix 가져오기 (캐시 사용)
            correlation = self._get_correlation_at_date(end_idx)

            return momentum, correlation

        return signal_provider

    def add_strategy(self, name: str, strategy_selector: Callable, use_inverse: bool = False, needs_macd: bool = False) -> None:
        """
        전략 추가 및 백테스트 실행

        Parameters:
        -----------
        name : str
            전략 이름
        strategy_selector : function
            포트폴리오 선택 함수
        use_inverse : bool
            인버스 ETF 사용 여부
        needs_macd : bool
            MACD 계산 필요 여부 (기본값: False, 성능 최적화)
        """
        print(f"\n{'='*70}")
        print(f"{name.upper()} 백테스트")
        print("="*70)

        # Pre-computed 데이터를 사용하는 시그널 제공자 생성
        signal_provider = self._create_signal_provider(needs_macd)

        inverse_prices = self.inverse_etf_prices if use_inverse else None
        values, returns, holdings = run_backtest(
            self.closeM,
            strategy_selector,
            inverse_prices,
            self.end_date,
            verbose=True,
            signal_provider=signal_provider
        )

        # 벤치마크 수익률 정렬
        benchmark_for_metrics = None
        if self.benchmark_returns is not None:
            benchmark_for_metrics = self.benchmark_returns.reindex(returns.index)

        metrics = calculate_metrics(returns, benchmark_for_metrics)

        self.strategies[name] = {
            'values': values,
            'returns': returns,
            'holdings': holdings,
            'metrics': metrics
        }

    def save_results(self, generate_report: bool = True) -> pd.DataFrame:
        """
        결과 저장 및 metrics DataFrame 반환

        Parameters:
        -----------
        generate_report : bool
            HTML 리포트 생성 여부

        Returns:
        --------
        pd.DataFrame
            metrics DataFrame
        """
        # benchmark를 전략과 동일한 기간으로 필터링
        filtered_benchmark = None
        if self.benchmark_returns is not None and self.strategies:
            first_strategy = list(self.strategies.values())[0]
            strategy_start = first_strategy['returns'].index[0]
            strategy_end = first_strategy['returns'].index[-1]

            # 시작일 이후 데이터만 필터링
            filtered_benchmark = self.benchmark_returns[
                (self.benchmark_returns.index > strategy_start) &
                (self.benchmark_returns.index <= strategy_end)
            ]

            # 시작일에 0.0 return 추가
            if len(filtered_benchmark) > 0:
                start_series = pd.Series([0.0], index=[strategy_start])
                filtered_benchmark = pd.concat([start_series, filtered_benchmark])

        # 필터링된 benchmark로 저장
        save_backtest_results(
            self.output_dir,
            self.strategies,
            filtered_benchmark
        )

        # metrics DataFrame 생성하여 반환
        metric_names = ['Total Return', 'CAGR', 'Volatility', 'Sharpe Ratio',
                        'Sortino Ratio', 'MDD', 'Win Rate', 'N Periods']

        metrics_data = {}
        for name, data in self.strategies.items():
            metrics_data[name] = [data['metrics'].get(m, np.nan) for m in metric_names]

        if filtered_benchmark is not None and len(filtered_benchmark) > 0:
            benchmark_metrics = calculate_metrics(filtered_benchmark)
            metrics_data['benchmark'] = [benchmark_metrics.get(m, np.nan) for m in metric_names]
            column_order = ['benchmark'] + [k for k in self.strategies.keys()]
            metrics_df = pd.DataFrame(metrics_data, index=metric_names)[column_order]
        else:
            metrics_df = pd.DataFrame(metrics_data, index=metric_names)

        # HTML 리포트 생성
        if generate_report:
            report_path = f'{self.output_dir}/report.html'
            generate_html_report(
                report_path,
                metrics_df,
                self.strategies,
                filtered_benchmark
            )

        return metrics_df


# ============================================================
# HTML 리포트 생성
# ============================================================

def generate_html_report(
    output_path: str,
    metrics_df: pd.DataFrame,
    strategies: Dict[str, Dict],
    benchmark_returns: Optional[pd.Series] = None
) -> None:
    """
    HTML 리포트 생성 (jinja2 템플릿 사용)

    Parameters:
    -----------
    output_path : str
        HTML 파일 저장 경로
    metrics_df : pd.DataFrame
        성과 지표 DataFrame
    strategies : dict
        전략별 데이터 {'strategy_name': {'returns': Series, 'values': Series}}
    benchmark_returns : pd.Series, optional
        벤치마크 수익률
    """
    from core.renderer import render_html_from_template
    from datetime import datetime

    # 1. 누적 수익률 차트
    fig_cumulative = go.Figure()

    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # 누적 수익률 계산 (시작일에 0.0이 이미 포함되어 있음)
        benchmark_values = (1 + benchmark_returns).cumprod()
        fig_cumulative.add_trace(go.Scatter(
            x=benchmark_values.index,
            y=benchmark_values.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='gray', width=2, dash='dash')
        ))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for idx, (name, data) in enumerate(strategies.items()):
        # portfolio values는 이미 누적된 값 (시작일에 1.0 포함)
        fig_cumulative.add_trace(go.Scatter(
            x=data['values'].index,
            y=data['values'].values,
            mode='lines',
            name=name.upper(),
            line=dict(color=colors[idx % len(colors)], width=2.5)
        ))

    fig_cumulative.update_layout(
        title='Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Portfolio Value',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )

    # 2. Drawdown 차트
    fig_drawdown = go.Figure()

    for idx, (name, data) in enumerate(strategies.items()):
        values = data['values']
        cummax = values.cummax()
        drawdown = (values - cummax) / cummax

        fig_drawdown.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode='lines',
            name=name.upper(),
            line=dict(color=colors[idx % len(colors)], width=2),
            fill='tozeroy'
        ))

    fig_drawdown.update_layout(
        title='Drawdown (%)',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    # 3. 월별 수익률 차트
    fig_monthly = go.Figure()

    if benchmark_returns is not None:
        fig_monthly.add_trace(go.Bar(
            x=benchmark_returns.index,
            y=benchmark_returns.values * 100,
            name='Benchmark',
            marker_color='lightgray',
            opacity=0.6
        ))

    for idx, (name, data) in enumerate(strategies.items()):
        returns = data['returns']
        fig_monthly.add_trace(go.Scatter(
            x=returns.index,
            y=returns.values * 100,
            mode='lines+markers',
            name=name.upper(),
            line=dict(color=colors[idx % len(colors)], width=2),
            marker=dict(size=4)
        ))

    fig_monthly.update_layout(
        title='Monthly Returns (%)',
        xaxis_title='Date',
        yaxis_title='Return (%)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    # 4. 연도별 수익률 바차트
    yearly_returns = {}
    for name, data in strategies.items():
        returns = data['returns']
        yearly = returns.groupby(returns.index.year).apply(lambda x: (1 + x).prod() - 1)
        yearly_returns[name.upper()] = yearly

    fig_yearly = go.Figure()
    years = sorted(set().union(*[set(yr.index) for yr in yearly_returns.values()]))

    for idx, (name, yearly) in enumerate(yearly_returns.items()):
        fig_yearly.add_trace(go.Bar(
            x=[str(y) for y in yearly.index],
            y=yearly.values * 100,
            name=name,
            marker_color=colors[idx % len(colors)]
        ))

    fig_yearly.update_layout(
        title='Annual Returns (%)',
        xaxis_title='Year',
        yaxis_title='Return (%)',
        barmode='group',
        template='plotly_white',
        height=400
    )

    # 5. 리스크-수익률 산점도
    fig_risk_return = go.Figure()

    risk_return_data = []
    for name, row in metrics_df.iterrows():
        if name != 'N Periods':
            continue
        break

    for col in metrics_df.columns:
        cagr = metrics_df.loc['CAGR', col] * 100
        vol = metrics_df.loc['Volatility', col] * 100
        sharpe = metrics_df.loc['Sharpe Ratio', col]

        color = 'gray' if col == 'benchmark' else colors[list(strategies.keys()).index(col) % len(colors)]
        marker_size = 12 if col == 'benchmark' else 15

        fig_risk_return.add_trace(go.Scatter(
            x=[vol],
            y=[cagr],
            mode='markers+text',
            name=col.upper(),
            text=[col.upper()],
            textposition='top center',
            marker=dict(size=marker_size, color=color),
            showlegend=True
        ))

    fig_risk_return.update_layout(
        title='Risk-Return Profile',
        xaxis_title='Volatility (% p.a.)',
        yaxis_title='CAGR (% p.a.)',
        template='plotly_white',
        height=500
    )

    # Plotly 차트를 HTML로 변환
    figures_html = [
        pio.to_html(fig_cumulative, include_plotlyjs=False, full_html=False),
        pio.to_html(fig_drawdown, include_plotlyjs=False, full_html=False),
        pio.to_html(fig_risk_return, include_plotlyjs=False, full_html=False),
        pio.to_html(fig_monthly, include_plotlyjs=False, full_html=False),
        pio.to_html(fig_yearly, include_plotlyjs=False, full_html=False),
    ]

    # Metrics 테이블을 HTML로 변환
    metrics_html = metrics_df.to_html(
        classes='table',
        float_format=lambda x: f'{x:.4f}',
        border=0
    )

    # Monthly Returns 테이블 (전체 기간)
    returns_df = pd.DataFrame({
        name.upper(): data['returns'] * 100
        for name, data in strategies.items()
    })
    if benchmark_returns is not None:
        returns_df.insert(0, 'BENCHMARK', benchmark_returns.reindex(returns_df.index).fillna(0) * 100)

    monthly_returns_html = returns_df.to_html(
        classes='table',
        float_format=lambda x: f'{x:.2f}%',
        border=0
    )

    # 템플릿 렌더링 데이터
    render_data = {
        'title': 'Backtest Report',
        'subtitle': f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        'metrics_html': metrics_html,
        'monthly_returns_html': monthly_returns_html,
        'figures': figures_html
    }

    # 템플릿을 사용하여 HTML 생성
    render_html_from_template('backtest_report.html', render_data, output_path)

    print(f"      HTML 리포트 생성 완료: {output_path}")


# ============================================================
# Strategy Configuration 패턴
# ============================================================

@dataclass
class StrategyConfig:
    """
    백테스트 전략 설정

    Attributes:
    -----------
    name : str
        전략 이름 (예: 'strategy1')
    momentum_ratio : float
        모멘텀 상위 비율 (0.33 = 상위 1/3, 0.5 = 상위 1/2)
    rsquared_ratio : float
        R-squared 상위 비율
    correlation_ratio : float
        Correlation 하위 비율 (낮을수록 분산 효과 강화)
    use_inverse : bool
        음수 모멘텀 종목에 인버스 ETF 사용 여부
    use_macd_filter : bool
        MACD 오실레이터 필터 사용 여부 (3MR - 12MR)
    description : str
        전략 설명
    """
    name: str
    momentum_ratio: float
    rsquared_ratio: float
    correlation_ratio: float
    use_inverse: bool = False
    use_macd_filter: bool = False
    description: str = ""


def create_strategy_selector(config: StrategyConfig) -> Callable:
    """
    전략 설정에 따라 포트폴리오 선택 함수를 생성하는 Factory 함수

    Parameters:
    -----------
    config : StrategyConfig
        전략 설정

    Returns:
    --------
    Callable
        (momentum, correlation) -> portfolio 형태의 선택 함수
    """
    def selector(momentum: pd.DataFrame, correlation: pd.DataFrame) -> Dict[str, float]:
        """전략 설정에 따른 포트폴리오 선택"""
        # 필터링
        tickers = apply_filters(
            momentum,
            correlation,
            config.momentum_ratio,
            config.rsquared_ratio,
            config.correlation_ratio
        )

        # 포트폴리오 구성
        portfolio = build_portfolio(
            tickers,
            momentum,
            use_inverse=config.use_inverse,
            use_macd_filter=config.use_macd_filter
        )

        return portfolio

    return selector


# ============================================================
# 전략 공통 로직
# ============================================================

def calculate_avg_momentum(momentum: pd.DataFrame) -> pd.Series:
    """
    13612MR 계산

    Parameters:
    -----------
    momentum : pd.DataFrame
        모멘텀 데이터프레임

    Returns:
    --------
    pd.Series
        13612MR (1MR + 3MR + 6MR + 12MR) / 4
    """
    return momentum['13612MR']


def calculate_avg_rsquared(momentum: pd.DataFrame) -> pd.Series:
    """
    mean-R² 계산: (1 + √RS3 + √RS6 + √RS12) / 4

    Parameters:
    -----------
    momentum : pd.DataFrame
        모멘텀 데이터프레임 (RS3, RS6, RS12 컬럼 필요)

    Returns:
    --------
    pd.Series
        평균 R-squared 값
    """
    return (1 + np.sqrt(momentum['RS3']) +
            np.sqrt(momentum['RS6']) +
            np.sqrt(momentum['RS12'])) / 4


def calculate_marginal_means(correlation: pd.DataFrame, tickers: List[str]) -> pd.Series:
    """
    Correlation marginal mean 계산

    Parameters:
    -----------
    correlation : pd.DataFrame
        상관계수 행렬
    tickers : List[str]
        대상 종목 리스트

    Returns:
    --------
    pd.Series
        각 종목의 marginal mean (다른 종목들과의 평균 상관계수)
    """
    # 'mean' 행/열 제거
    corr_matrix = correlation.drop('mean', axis=0, errors='ignore').drop('mean', axis=1, errors='ignore')

    # 해당 종목들만 필터링
    available_tickers = [t for t in tickers if t in corr_matrix.index]

    if len(available_tickers) == 0:
        return pd.Series(dtype=float)

    sub_corr = corr_matrix.loc[available_tickers, available_tickers]
    n = len(available_tickers)

    if n > 1:
        # 대각선 제외한 평균
        marginal_means = (sub_corr.sum(axis=1) - 1) / (n - 1)
        return marginal_means
    else:
        # 종목이 1개면 marginal mean 없음
        return pd.Series([0.0], index=available_tickers)


def apply_filters(
    momentum: pd.DataFrame,
    correlation: pd.DataFrame,
    momentum_ratio: float,
    rsquared_ratio: float,
    correlation_ratio: float
) -> List[str]:
    """
    3단계 필터링 수행

    Parameters:
    -----------
    momentum : pd.DataFrame
        모멘텀 데이터
    correlation : pd.DataFrame
        상관계수 행렬
    momentum_ratio : float
        모멘텀 상위 비율 (예: 0.5 = 상위 50%)
    rsquared_ratio : float
        R-squared 상위 비율
    correlation_ratio : float
        Correlation 하위 비율

    Returns:
    --------
    List[str]
        필터링된 종목 리스트
    """
    # NaN 제거
    momentum = momentum.dropna(subset=['13612MR', 'RS3', 'RS6', 'RS12'])
    if len(momentum) == 0:
        return []

    total_stocks = len(momentum)

    # Step 1: 평균 모멘텀 필터링
    avg_momentum = calculate_avg_momentum(momentum)
    step1_count = max(1, int(total_stocks * momentum_ratio))
    step1_tickers = avg_momentum.nlargest(step1_count).index.tolist()

    # Step 2: 평균 R-squared 필터링
    avg_rsquared = calculate_avg_rsquared(momentum)
    step2_candidates = avg_rsquared[step1_tickers]
    step2_count = max(1, int(len(step1_tickers) * rsquared_ratio))
    step2_tickers = step2_candidates.nlargest(step2_count).index.tolist()

    # Step 3: Marginal mean 필터링
    marginal_means = calculate_marginal_means(correlation, step2_tickers)
    if len(marginal_means) == 0:
        return []

    step3_count = max(1, int(len(marginal_means) * correlation_ratio))
    step3_tickers = marginal_means.nsmallest(step3_count).index.tolist()

    # DEBUG: 필터링 결과 출력 (첫 번째 호출 시에만)
    if not hasattr(apply_filters, '_debug_printed'):
        print(f"\n[DEBUG] 필터링 단계별 종목 수:")
        print(f"  전체: {total_stocks}개")
        print(f"  Step 1 (모멘텀 {momentum_ratio:.1%}): {len(step1_tickers)}개")
        print(f"  Step 2 (R² {rsquared_ratio:.1%}): {len(step2_tickers)}개")
        print(f"  Step 3 (상관 {correlation_ratio:.1%}): {len(step3_tickers)}개")
        print(f"  계산식: {total_stocks} × {momentum_ratio} × {rsquared_ratio} × {correlation_ratio} = {total_stocks * momentum_ratio * rsquared_ratio * correlation_ratio:.1f}\n")
        apply_filters._debug_printed = True

    return step3_tickers


def build_portfolio(
    tickers: List[str],
    momentum: pd.DataFrame,
    use_inverse: bool = False,
    use_macd_filter: bool = False
) -> Dict[str, float]:
    """
    포트폴리오 구성

    Parameters:
    -----------
    tickers : List[str]
        선택된 종목 리스트
    momentum : pd.DataFrame
        모멘텀 데이터
    use_inverse : bool
        음수 모멘텀 종목에 인버스 ETF 사용 여부
    use_macd_filter : bool
        MACD 필터 사용 여부

    Returns:
    --------
    Dict[str, float]
        포트폴리오 (ticker: weight)
    """
    if len(tickers) == 0:
        return {}

    n_stocks = len(tickers)
    equal_weight = 1.0 / n_stocks if n_stocks > 0 else 0

    # 벡터 연산으로 필터링
    mom_values = momentum.loc[tickers, '13612MR']

    if use_macd_filter:
        # MACD 필터 적용 (벡터 연산)
        macd_values = momentum.loc[tickers, 'MACD_Histogram']
        # 13612MR >= 0 AND MACD_Histogram >= 0
        valid_mask = (mom_values >= 0) & (macd_values >= 0)
        valid_tickers = mom_values[valid_mask].index.tolist()
        portfolio = {ticker: equal_weight for ticker in valid_tickers}
        inverse_weight = 0.0

        # DEBUG
        if not hasattr(build_portfolio, '_debug_printed'):
            print(f"[DEBUG] 포트폴리오 필터링:")
            print(f"  선택된 종목: {n_stocks}개")
            print(f"  MACD+모멘텀 필터 통과: {len(valid_tickers)}개")
            print(f"  투자 비중: {len(valid_tickers) * equal_weight:.1%}\n")
            build_portfolio._debug_printed = True

    else:
        # 기본 필터만 적용
        positive_mask = mom_values >= 0
        negative_mask = mom_values < 0

        # 양수 모멘텀: 투자
        valid_tickers = mom_values[positive_mask].index.tolist()
        portfolio = {ticker: equal_weight for ticker in valid_tickers}

        # 음수 모멘텀: 인버스 또는 현금
        inverse_weight = 0.0
        if use_inverse:
            n_negative = negative_mask.sum()
            inverse_weight = (equal_weight / 4) * n_negative

        # DEBUG
        if not hasattr(build_portfolio, '_debug_printed'):
            print(f"[DEBUG] 포트폴리오 필터링:")
            print(f"  선택된 종목: {n_stocks}개")
            print(f"  모멘텀 >= 0: {len(valid_tickers)}개")
            print(f"  모멘텀 < 0: {negative_mask.sum()}개")
            print(f"  투자 비중: {len(valid_tickers) * equal_weight:.1%}\n")
            build_portfolio._debug_printed = True

    # 인버스 가중치 추가
    if inverse_weight > 0 and use_inverse:
        portfolio['INVERSE'] = inverse_weight

    return portfolio
