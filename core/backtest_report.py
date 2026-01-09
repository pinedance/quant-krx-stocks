"""
백테스트 리포트 생성 모듈

HTML 리포트 및 차트 생성 함수들을 제공합니다.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from typing import Optional, Dict
from datetime import datetime


# ============================================================
# Chart 색상 팔레트
# ============================================================

CHART_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


# ============================================================
# 개별 차트 생성 함수
# ============================================================

def _create_cumulative_returns_chart(strategies: Dict[str, Dict], benchmark_returns: Optional[pd.Series] = None):
    """누적 수익률 차트 생성"""
    fig = go.Figure()

    if benchmark_returns is not None and len(benchmark_returns) > 0:
        benchmark_values = (1 + benchmark_returns).cumprod()
        fig.add_trace(go.Scatter(
            x=benchmark_values.index,
            y=benchmark_values.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='gray', width=2, dash='dash')
        ))

    for idx, (name, data) in enumerate(strategies.items()):
        fig.add_trace(go.Scatter(
            x=data['values'].index,
            y=data['values'].values,
            mode='lines',
            name=name.upper(),
            line=dict(color=CHART_COLORS[idx % len(CHART_COLORS)], width=2.5)
        ))

    fig.update_layout(
        title='Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Portfolio Value',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    return fig


def _create_drawdown_chart(strategies: Dict[str, Dict]):
    """Drawdown 차트 생성"""
    fig = go.Figure()

    for idx, (name, data) in enumerate(strategies.items()):
        values = data['values']
        cummax = values.cummax()
        drawdown = (values - cummax) / cummax

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode='lines',
            name=name.upper(),
            line=dict(color=CHART_COLORS[idx % len(CHART_COLORS)], width=2),
            fill='tozeroy'
        ))

    fig.update_layout(
        title='Drawdown (%)',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    return fig


def _create_monthly_returns_chart(strategies: Dict[str, Dict], benchmark_returns: Optional[pd.Series] = None):
    """월별 수익률 차트 생성"""
    fig = go.Figure()

    if benchmark_returns is not None:
        fig.add_trace(go.Bar(
            x=benchmark_returns.index,
            y=benchmark_returns.values * 100,
            name='Benchmark',
            marker_color='lightgray',
            opacity=0.6
        ))

    for idx, (name, data) in enumerate(strategies.items()):
        returns = data['returns']
        fig.add_trace(go.Scatter(
            x=returns.index,
            y=returns.values * 100,
            mode='lines+markers',
            name=name.upper(),
            line=dict(color=CHART_COLORS[idx % len(CHART_COLORS)], width=2),
            marker=dict(size=4)
        ))

    fig.update_layout(
        title='Monthly Returns (%)',
        xaxis_title='Date',
        yaxis_title='Return (%)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    return fig


def _create_annual_returns_chart(strategies: Dict[str, Dict]):
    """연도별 수익률 차트 생성"""
    yearly_returns = {}
    for name, data in strategies.items():
        returns = data['returns']
        yearly = returns.groupby(returns.index.year).apply(lambda x: (1 + x).prod() - 1)
        yearly_returns[name.upper()] = yearly

    fig = go.Figure()
    for idx, (name, yearly) in enumerate(yearly_returns.items()):
        fig.add_trace(go.Bar(
            x=[str(y) for y in yearly.index],
            y=yearly.values * 100,
            name=name,
            marker_color=CHART_COLORS[idx % len(CHART_COLORS)]
        ))

    fig.update_layout(
        title='Annual Returns (%)',
        xaxis_title='Year',
        yaxis_title='Return (%)',
        barmode='group',
        template='plotly_white',
        height=400
    )
    return fig


def _create_risk_return_chart(metrics_df: pd.DataFrame, strategies: Dict[str, Dict]):
    """리스크-수익률 산점도 생성"""
    fig = go.Figure()

    for col in metrics_df.columns:
        cagr = metrics_df.loc['CAGR', col] * 100
        vol = metrics_df.loc['Volatility', col] * 100

        is_benchmark = col == 'benchmark'
        color = 'gray' if is_benchmark else CHART_COLORS[list(strategies.keys()).index(col) % len(CHART_COLORS)]
        marker_size = 12 if is_benchmark else 15

        fig.add_trace(go.Scatter(
            x=[vol],
            y=[cagr],
            mode='markers+text',
            name=col.upper(),
            text=[col.upper()],
            textposition='top center',
            marker=dict(size=marker_size, color=color),
            showlegend=True
        ))

    fig.update_layout(
        title='Risk-Return Profile',
        xaxis_title='Volatility (% p.a.)',
        yaxis_title='CAGR (% p.a.)',
        template='plotly_white',
        height=500
    )
    return fig


# ============================================================
# HTML 리포트 생성
# ============================================================

def generate_html_report(
    output_path: str,
    metrics_df: pd.DataFrame,
    strategies: Dict[str, Dict],
    benchmark_returns: Optional[pd.Series] = None,
    final_portfolios: Optional[pd.DataFrame] = None
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
    final_portfolios : pd.DataFrame, optional
        최종 포트폴리오 (Strategy, Ticker, Name, Weight 컬럼)
    """
    from core.renderer import render_template
    from core.file import save_html

    # 1. 누적 수익률 차트
    fig_cumulative = _create_cumulative_returns_chart(strategies, benchmark_returns)

    # 2. Drawdown 차트
    fig_drawdown = _create_drawdown_chart(strategies)

    # 3. 월별 수익률 차트
    fig_monthly = _create_monthly_returns_chart(strategies, benchmark_returns)

    # 4. 연도별 수익률 바차트
    fig_yearly = _create_annual_returns_chart(strategies)

    # 5. 리스크-수익률 산점도
    fig_risk_return = _create_risk_return_chart(metrics_df, strategies)

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

    # 최종 포트폴리오 탭별 HTML 생성
    portfolio_tabs = []
    if final_portfolios is not None and len(final_portfolios) > 0:
        for strategy_name in strategies.keys():
            strategy_portfolio = final_portfolios[final_portfolios['Strategy'] == strategy_name].copy()
            # Strategy 컬럼 제거하고 HTML 생성
            strategy_portfolio = strategy_portfolio[['Ticker', 'Name', 'Weight']]
            # Weight를 퍼센트로 포맷
            portfolio_html = strategy_portfolio.to_html(
                classes='table',
                index=False,
                border=0,
                formatters={'Weight': lambda x: f'{x*100:.2f}%'}
            )
            portfolio_tabs.append({
                'name': strategy_name.upper(),
                'content': portfolio_html
            })

    # 템플릿 렌더링 데이터
    render_data = {
        'title': 'Backtest Report',
        'subtitle': f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        'metrics_html': metrics_html,
        'monthly_returns_html': monthly_returns_html,
        'figures': figures_html,
        'portfolio_tabs': portfolio_tabs
    }

    # 템플릿을 사용하여 HTML 생성
    content = render_template('backtest_report.html', render_data)
    save_html(content, output_path)

    print(f"      HTML 리포트 생성 완료: {output_path}")
