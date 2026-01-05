"""
백테스트 실행 모듈

백테스트 실행, ETF 데이터 로드, 성과 지표 계산, 결과 저장 함수들을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Callable, Optional, Dict, Tuple, List
from core.finance import calculate_corr_matrix
from core.signals import calculate_all_momentum, calculate_all_macd, calculate_signals_at_date
from core.fetcher import get_etf_data
from core.config import settings


# ============================================================
# 상수 정의
# ============================================================

# 백테스트 설정
MIN_BACKTEST_MONTHS = 12  # 백테스트 시작 전 필요한 최소 데이터 개월 수
INVERSE_WEIGHT_RATIO = 0.25  # 인버스 ETF 비중 비율 (1/4)
MONTHS_PER_YEAR = 12  # 연간 개월 수


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
    n_years = len(returns) / MONTHS_PER_YEAR
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # MDD (Maximum Drawdown)
    cummax = cumulative_returns.cummax()
    drawdown = (cumulative_returns - cummax) / cummax
    mdd = drawdown.min()

    # Volatility (연율화)
    volatility = returns.std() * np.sqrt(MONTHS_PER_YEAR)

    # Sharpe Ratio (무위험 수익률 0 가정)
    sharpe = cagr / volatility if volatility > 0 else 0

    # Sortino Ratio (하방 편차만 고려)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(MONTHS_PER_YEAR) if len(downside_returns) > 0 else 0
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
        tracking_error = excess_returns.std() * np.sqrt(MONTHS_PER_YEAR)
        metrics['Tracking Error'] = tracking_error

        # Information Ratio
        excess_return = excess_returns.mean() * MONTHS_PER_YEAR
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
        metrics['Information Ratio'] = information_ratio

        # Alpha (초과 수익)
        benchmark_cagr = (1 + (1 + benchmark_returns).prod() - 1) ** (1 / n_years) - 1 if n_years > 0 else 0
        metrics['Alpha'] = cagr - benchmark_cagr

    return metrics


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
    start_idx = MIN_BACKTEST_MONTHS + 1

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

    # Index 범위 체크: i+1이 유효한 범위 내에 있도록 보장
    max_idx = min(end_idx, len(closeM) - 1)

    for i in range(start_idx, max_idx):
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
        if verbose and (i == start_idx or (i - start_idx + 1) % MONTHS_PER_YEAR == 1):
            if inverse_weight > 0:
                print(f"  {signal_date.strftime('%Y-%m')}: 포트폴리오 가치 = {current_value:.4f} "
                      f"({len(stock_portfolio)}개 종목, 인버스 {inverse_weight:.2%})")
            else:
                print(f"  {signal_date.strftime('%Y-%m')}: 포트폴리오 가치 = {current_value:.4f} "
                      f"({len(stock_portfolio)}개 종목)")

    # 최종 날짜(end_idx)의 시그널로 "미래 투자용" 포트폴리오 생성
    if end_idx < len(closeM):
        final_date = closeM.index[end_idx]
        final_momentum, final_correlation = signal_provider(end_idx)
        final_portfolio = strategy_selector(final_momentum, final_correlation)

        # Holdings 기록 (수익률 계산 없이 포트폴리오만)
        final_inverse_weight = final_portfolio.get('INVERSE', 0.0)
        final_stock_portfolio = {k: v for k, v in final_portfolio.items() if k != 'INVERSE'}

        holdings_history.append({
            'date': final_date,
            'holdings': final_stock_portfolio.copy(),
            'n_stocks': len(final_stock_portfolio),
            'inverse_weight': final_inverse_weight,
            'cash_weight': 1.0 - sum(final_portfolio.values())
        })

        if verbose:
            if final_inverse_weight > 0:
                print(f"\n  최종 포트폴리오 ({final_date.strftime('%Y-%m')}): "
                      f"{len(final_stock_portfolio)}개 종목, 인버스 {final_inverse_weight:.2%}")
            else:
                print(f"\n  최종 포트폴리오 ({final_date.strftime('%Y-%m')}): "
                      f"{len(final_stock_portfolio)}개 종목")

    return (pd.Series(portfolio_values, index=dates),
            pd.Series(monthly_returns, index=dates),
            holdings_history)


# ============================================================
# 결과 저장
# ============================================================

def _build_final_portfolios(strategies: Dict[str, Dict], tickers_info: pd.DataFrame) -> pd.DataFrame:
    """
    전략별 최종 포트폴리오를 하나의 DataFrame으로 구성

    Parameters:
    -----------
    strategies : dict
        전략별 데이터 ({'holdings': list, ...})
    tickers_info : pd.DataFrame
        종목 정보 (Code, Name 포함)

    Returns:
    --------
    pd.DataFrame
        Strategy, Ticker, Name, Weight 컬럼을 가진 DataFrame (Ticker 오름차순 정렬)
    """
    # 종목 코드 → 이름 매핑
    ticker_to_name = dict(zip(tickers_info['Code'], tickers_info['Name']))

    all_portfolios = []

    for strategy_name, data in strategies.items():
        holdings_history = data.get('holdings', [])
        if len(holdings_history) == 0:
            # 빈 포트폴리오: Cash 100%
            all_portfolios.append({
                'Strategy': strategy_name,
                'Ticker': 'Cash',
                'Name': '',
                'Weight': 1.0
            })
            continue

        # 최종(최근) 포트폴리오
        final_holding = holdings_history[-1]
        stock_holdings = final_holding.get('holdings', {})
        inverse_weight = final_holding.get('inverse_weight', 0.0)
        cash_weight = final_holding.get('cash_weight', 0.0)

        # 주식 종목 (Ticker 오름차순 정렬)
        for ticker in sorted(stock_holdings.keys()):
            weight = stock_holdings[ticker]
            all_portfolios.append({
                'Strategy': strategy_name,
                'Ticker': ticker,
                'Name': ticker_to_name.get(ticker, ''),
                'Weight': weight
            })

        # 인버스 ETF (주식 종목 다음)
        if inverse_weight > 0:
            all_portfolios.append({
                'Strategy': strategy_name,
                'Ticker': 'INVERSE',
                'Name': 'KODEX 인버스',
                'Weight': inverse_weight
            })

        # 현금 (맨 마지막)
        if cash_weight > 0:
            all_portfolios.append({
                'Strategy': strategy_name,
                'Ticker': 'Cash',
                'Name': '',
                'Weight': cash_weight
            })

    return pd.DataFrame(all_portfolios)


def save_backtest_results(
    output_dir: str,
    strategies: Dict[str, Dict],
    benchmark_returns: Optional[pd.Series] = None,
    tickers_info: Optional[pd.DataFrame] = None
) -> None:
    """
    백테스트 결과를 파일로 저장

    Parameters:
    -----------
    output_dir : str
        저장 디렉토리
    strategies : dict
        {'strategy_name': {'returns': Series, 'values': Series, 'holdings': list, 'metrics': dict}}
    benchmark_returns : pd.Series, optional
        벤치마크 수익률 (전략과 동일한 기간으로 필터링된 것)
    tickers_info : pd.DataFrame, optional
        종목 정보 (Code, Name 포함) - 최종 포트폴리오 저장 시 필요
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

    # 최종 포트폴리오 저장
    if tickers_info is not None:
        final_portfolios = _build_final_portfolios(strategies, tickers_info)
        if final_portfolios is not None and len(final_portfolios) > 0:
            portfolios_path = f'{output_dir}/portfolio_latest.tsv'
            final_portfolios.to_csv(portfolios_path, sep='\t', index=False)
            print(f"      portfolio latest 저장 완료: {portfolios_path}")


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
            correlation = calculate_corr_matrix(prices, corr_periods)
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

        # 종목 정보 로드 (최종 포트폴리오 저장용)
        tickers_info = None
        try:
            from core.file import import_dataframe_from_json
            from core.config import settings
            market = settings.stocks.list.market
            list_dir = settings.output.list_dir.path
            tickers_info = import_dataframe_from_json(f'{list_dir}/{market}.json')
        except Exception as e:
            print(f"      경고: 종목 정보 로드 실패 ({e}) - 최종 포트폴리오 종목명 없이 저장")

        # 필터링된 benchmark로 저장
        save_backtest_results(
            self.output_dir,
            self.strategies,
            filtered_benchmark,
            tickers_info
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
            from core.backtest_report import generate_html_report

            report_path = f'{self.output_dir}/report.html'
            # 최종 포트폴리오 구성
            final_portfolios = None
            if tickers_info is not None:
                final_portfolios = _build_final_portfolios(self.strategies, tickers_info)

            generate_html_report(
                report_path,
                metrics_df,
                self.strategies,
                filtered_benchmark,
                final_portfolios
            )

        return metrics_df
