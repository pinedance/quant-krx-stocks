"""
STEP 4: KRX300 대시보드 생성
- Momentum, Performance, Correlation 대시보드 생성
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import networkx as nx
from core.config import settings
from core.io import import_dataframe_from_json, render_dashboard_html, render_html_from_template


# ============================================================
# Constants
# ============================================================

MARGINAL_MEAN_COLUMN = 'mean'


# ============================================================
# Data Transformation Functions
# ============================================================

def calculate_momentum_quality(momentum, period):
    """
    Momentum Quality (R × AS) 계산

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 데이터
    period : int
        기간 (개월)

    Returns:
    --------
    pd.Series
        Momentum Quality 값
    """
    return np.sqrt(momentum[f'RS{period}']) * momentum[f'AS{period}']


def calculate_sharpe_ratio(performance, period):
    """
    Sharpe Ratio (AR / SD) 계산

    Parameters:
    -----------
    performance : pd.DataFrame
        Performance 데이터
    period : int
        기간 (개월)

    Returns:
    --------
    pd.Series
        Sharpe Ratio 값
    """
    return performance[f'AR{period}'] / performance[f'SD{period}']


def calculate_sortino_ratio(performance, period):
    """
    Sortino Ratio (AR / DD) 계산

    Parameters:
    -----------
    performance : pd.DataFrame
        Performance 데이터
    period : int
        기간 (개월)

    Returns:
    --------
    pd.Series
        Sortino Ratio 값
    """
    return performance[f'AR{period}'] / performance[f'DD{period}']


def calculate_correlation_coefficient(momentum, period):
    """
    Correlation Coefficient (√R²) 계산

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 데이터
    period : int
        기간 (개월)

    Returns:
    --------
    pd.Series
        Correlation Coefficient 값
    """
    return np.sqrt(momentum[f'RS{period}'])


# ============================================================
# Helper Functions: Axis Range Calculation
# ============================================================

def calculate_axis_range(data_series):
    """
    축 범위를 계산합니다.

    Parameters:
    -----------
    data_series : pd.Series
        축에 표시할 데이터

    Returns:
    --------
    list[float, float]
        [min, max] 범위
    """
    padding = settings.visualization.scatter_plot.axis_padding
    pos_mult = padding.positive_multiplier
    neg_mult = padding.negative_multiplier
    min_mult = padding.min_multiplier

    min_val = data_series.min()
    max_val = data_series.max()

    range_min = min_val * neg_mult if min_val < 0 else min_val * min_mult
    range_max = max_val * pos_mult if max_val > 0 else max_val * min_mult

    return [range_min, range_max]


# ============================================================
# Helper Functions: Trendline
# ============================================================

def add_trendline(fig, x_data, y_data, color, name=None, show_in_legend=True):
    """
    산점도에 추세선을 추가합니다.

    Parameters:
    -----------
    fig : go.Figure
        Plotly Figure 객체
    x_data : pd.Series
        X축 데이터
    y_data : pd.Series
        Y축 데이터
    color : str
        추세선 색상
    name : str, optional
        범례 이름
    show_in_legend : bool
        범례 표시 여부
    """
    style = settings.visualization.scatter_plot.trendline

    # NaN 제거
    non_null_mask = ~(x_data.isna() | y_data.isna())
    if non_null_mask.sum() <= 1:
        return

    x_valid = x_data[non_null_mask]
    y_valid = y_data[non_null_mask]

    # 선형 회귀
    coeffs = np.polyfit(x_valid, y_valid, 1)
    poly_func = np.poly1d(coeffs)

    # 추세선 좌표
    x_trend = np.array([x_valid.min(), x_valid.max()])
    y_trend = poly_func(x_trend)

    # 이름 생성
    if name is None:
        name = f'추세선 (y={coeffs[0]:.2f}x+{coeffs[1]:.2f})'

    fig.add_trace(
        go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            line=dict(color=color, dash=style.dash, width=style.width),
            opacity=style.opacity,
            name=name,
            showlegend=show_in_legend,
            hoverinfo='skip'
        )
    )


# ============================================================
# Helper Functions: Scatter Trace
# ============================================================

def add_scatter_trace(fig, x_data, y_data, labels, color, name, hover_template):
    """
    산점도 trace를 추가합니다.

    Parameters:
    -----------
    fig : go.Figure
        Plotly Figure 객체
    x_data : pd.Series
        X축 데이터
    y_data : pd.Series
        Y축 데이터
    labels : pd.Index
        데이터 포인트 라벨 (ticker 이름 등)
    color : str
        마커 색상
    name : str
        범례 이름
    hover_template : str
        호버 템플릿
    """
    marker_style = settings.visualization.scatter_plot.marker

    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            marker=dict(size=marker_style.size, color=color, opacity=marker_style.opacity),
            text=labels,
            hovertemplate=hover_template,
            name=name
        )
    )


# ============================================================
# Helper Functions: Reference Lines
# ============================================================

def add_reference_lines(fig, add_horizontal=True, add_vertical=False):
    """
    참조선(0을 나타내는 선)을 추가합니다.

    Parameters:
    -----------
    fig : go.Figure
        Plotly Figure 객체
    add_horizontal : bool
        수평선 추가 여부
    add_vertical : bool
        수직선 추가 여부
    """
    ref_style = settings.visualization.scatter_plot.reference_line

    if add_horizontal:
        fig.add_hline(y=0, line_dash=ref_style.dash, line_color=ref_style.color)
    if add_vertical:
        fig.add_vline(x=0, line_dash=ref_style.dash, line_color=ref_style.color)


# ============================================================
# Helper Functions: Base Figure
# ============================================================

def create_base_figure(title, x_title, y_title, x_range, y_range):
    """
    기본 Figure 객체를 생성하고 레이아웃을 설정합니다.

    Parameters:
    -----------
    title : str
        차트 제목
    x_title : str
        X축 제목
    y_title : str
        Y축 제목
    x_range : list[float, float]
        X축 범위
    y_range : list[float, float]
        Y축 범위

    Returns:
    --------
    go.Figure
        설정된 Figure 객체
    """
    layout_style = settings.visualization.scatter_plot.layout
    height = settings.dashboard.height

    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis_range=x_range,
        yaxis_range=y_range,
        height=height,
        showlegend=layout_style.showlegend,
        hovermode=layout_style.hovermode
    )

    return fig


# ============================================================
# Helper Functions: Multi-Period Chart Creation
# ============================================================

def create_multi_period_scatter_chart(
    title, x_title, y_title,
    x_data_func, y_data_func,
    hover_template_func,
    labels,
    periods=None,
    colors=None,
    add_horizontal_ref=True,
    add_vertical_ref=False
):
    """
    여러 기간 데이터를 표시하는 산점도 차트를 생성합니다.

    Parameters:
    -----------
    title : str
        차트 제목
    x_title : str
        X축 제목
    y_title : str
        Y축 제목
    x_data_func : callable
        period를 받아 해당 기간의 X축 데이터를 반환하는 함수
    y_data_func : callable
        period를 받아 해당 기간의 Y축 데이터를 반환하는 함수
    hover_template_func : callable
        period를 받아 hover template를 반환하는 함수
    labels : pd.Index
        데이터 포인트 라벨 (ticker 이름 등)
    periods : list, optional
        표시할 기간 리스트. None이면 settings에서 가져옴
    colors : list, optional
        각 기간에 사용할 색상 리스트. None이면 settings에서 가져옴
    add_horizontal_ref : bool
        수평 참조선 추가 여부
    add_vertical_ref : bool
        수직 참조선 추가 여부

    Returns:
    --------
    go.Figure
        생성된 차트
    """
    if periods is None:
        periods = settings.visualization.scatter_plot.periods
    if colors is None:
        colors = settings.visualization.scatter_plot.colors

    # 축 범위 계산
    all_x_data = [x_data_func(period) for period in periods]
    all_y_data = [y_data_func(period) for period in periods]

    x_range = calculate_axis_range(pd.concat(all_x_data))
    y_range = calculate_axis_range(pd.concat(all_y_data))

    # Figure 생성
    fig = create_base_figure(title, x_title, y_title, x_range, y_range)

    # 각 기간별 trace 추가
    for period, color in zip(periods, colors):
        x_data = x_data_func(period)
        y_data = y_data_func(period)

        add_scatter_trace(
            fig=fig,
            x_data=x_data,
            y_data=y_data,
            labels=labels,
            color=color,
            name=f'{period}개월',
            hover_template=hover_template_func(period)
        )

        add_trendline(
            fig=fig,
            x_data=x_data,
            y_data=y_data,
            color=color,
            name=f'{period}M 추세선',
            show_in_legend=False
        )

    # 참조선
    add_reference_lines(fig, add_horizontal_ref, add_vertical_ref)

    return fig


# ============================================================
# Chart Creation: Momentum
# ============================================================

def create_monthly_momentum_chart(momentum):
    """
    Monthly Momentum 차트를 생성합니다 (R vs 13612MR, 12개월 기준).

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 데이터

    Returns:
    --------
    go.Figure
        생성된 차트
    """
    # 팔레트에서 색상 가져오기 (12개월이므로 첫 번째 색상)
    palette = settings.visualization.scatter_plot.colors

    return create_multi_period_scatter_chart(
        title="Monthly Momentum",
        x_title="Correlation Coefficient (R, 12M)",
        y_title="평균 모멘텀 (13612MR)",
        x_data_func=lambda p: calculate_correlation_coefficient(momentum, p),
        y_data_func=lambda p: momentum['13612MR'],
        hover_template_func=lambda p: '<b>%{text}</b><br>R (12M): %{x:.3f}<br>13612MR: %{y:.2%}<extra></extra>',
        labels=momentum.index,
        periods=[12],
        colors=[palette[0]],
        add_horizontal_ref=True,
        add_vertical_ref=True
    )


def create_regression_momentum_chart(momentum):
    """
    Regression Momentum 차트를 생성합니다 (R² vs Annualized Slope).

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 데이터

    Returns:
    --------
    go.Figure
        생성된 차트
    """
    return create_multi_period_scatter_chart(
        title="Regression Momentum",
        x_title="R-squared",
        y_title="Annualized Slope",
        x_data_func=lambda p: momentum[f'RS{p}'],
        y_data_func=lambda p: momentum[f'AS{p}'],
        hover_template_func=lambda p: f'<b>%{{text}}</b><br>{p}M<br>R²: %{{x:.3f}}<br>Ann. Slope: %{{y:.2%}}<extra></extra>',
        labels=momentum.index,
        add_horizontal_ref=True,
        add_vertical_ref=False
    )


def create_momentum_quality_vs_return_chart(momentum, performance):
    """
    Momentum Quality vs Return 차트를 생성합니다 (R × AS vs AR).

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 데이터
    performance : pd.DataFrame
        Performance 데이터

    Returns:
    --------
    go.Figure
        생성된 차트
    """
    return create_multi_period_scatter_chart(
        title="Momentum Quality vs Return",
        x_title="Momentum Quality (R × Annualized Slope)",
        y_title="Annualized Return",
        x_data_func=lambda p: calculate_momentum_quality(momentum, p),
        y_data_func=lambda p: performance[f'AR{p}'],
        hover_template_func=lambda p: f'<b>%{{text}}</b><br>{p}M<br>Quality: %{{x:.3f}}<br>AR: %{{y:.2%}}<extra></extra>',
        labels=momentum.index,
        add_horizontal_ref=True,
        add_vertical_ref=False
    )


def create_momentum_strength_vs_return_chart(momentum, performance):
    """
    Momentum Strength vs Return 차트를 생성합니다 (AS vs AR).

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 데이터
    performance : pd.DataFrame
        Performance 데이터

    Returns:
    --------
    go.Figure
        생성된 차트
    """
    return create_multi_period_scatter_chart(
        title="Momentum Strength vs Return",
        x_title="Annualized Slope",
        y_title="Annualized Return",
        x_data_func=lambda p: momentum[f'AS{p}'],
        y_data_func=lambda p: performance[f'AR{p}'],
        hover_template_func=lambda p: f'<b>%{{text}}</b><br>{p}M<br>AS: %{{x:.2%}}<br>AR: %{{y:.2%}}<extra></extra>',
        labels=momentum.index,
        add_horizontal_ref=True,
        add_vertical_ref=False
    )


def create_momentum_reliability_vs_return_chart(momentum, performance):
    """
    Momentum Reliability vs Return 차트를 생성합니다 (R vs AR).

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 데이터
    performance : pd.DataFrame
        Performance 데이터

    Returns:
    --------
    go.Figure
        생성된 차트
    """
    return create_multi_period_scatter_chart(
        title="Momentum Reliability vs Return",
        x_title="Correlation Coefficient (R)",
        y_title="Annualized Return",
        x_data_func=lambda p: calculate_correlation_coefficient(momentum, p),
        y_data_func=lambda p: performance[f'AR{p}'],
        hover_template_func=lambda p: f'<b>%{{text}}</b><br>{p}M<br>R: %{{x:.3f}}<br>AR: %{{y:.2%}}<extra></extra>',
        labels=momentum.index,
        add_horizontal_ref=True,
        add_vertical_ref=False
    )


# ============================================================
# Chart Creation: Performance
# ============================================================

def create_sharpe_ratio_chart(performance):
    """
    Sharpe Ratio 차트를 생성합니다 (SD vs AR).

    Parameters:
    -----------
    performance : pd.DataFrame
        Performance 데이터

    Returns:
    --------
    go.Figure
        생성된 차트
    """
    return create_multi_period_scatter_chart(
        title="Sharpe Ratio",
        x_title="Standard Deviation",
        y_title="Annualized Return",
        x_data_func=lambda p: performance[f'SD{p}'],
        y_data_func=lambda p: performance[f'AR{p}'],
        hover_template_func=lambda p: f'<b>%{{text}}</b><br>{p}M<br>SD: %{{x:.2%}}<br>AR: %{{y:.2%}}<extra></extra>',
        labels=performance.index,
        add_horizontal_ref=False,
        add_vertical_ref=False
    )


def create_sortino_ratio_chart(performance):
    """
    Sortino Ratio 차트를 생성합니다 (DD vs AR).

    Parameters:
    -----------
    performance : pd.DataFrame
        Performance 데이터

    Returns:
    --------
    go.Figure
        생성된 차트
    """
    return create_multi_period_scatter_chart(
        title="Sortino Ratio",
        x_title="Downside Deviation",
        y_title="Annualized Return",
        x_data_func=lambda p: performance[f'DD{p}'],
        y_data_func=lambda p: performance[f'AR{p}'],
        hover_template_func=lambda p: f'<b>%{{text}}</b><br>{p}M<br>DD: %{{x:.2%}}<br>AR: %{{y:.2%}}<extra></extra>',
        labels=performance.index,
        add_horizontal_ref=False,
        add_vertical_ref=False
    )


def create_momentum_quality_vs_sharpe_chart(momentum, performance):
    """
    Momentum Quality vs Sharpe Ratio 차트를 생성합니다 (R × AS vs AR/SD).

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 데이터
    performance : pd.DataFrame
        Performance 데이터

    Returns:
    --------
    go.Figure
        생성된 차트
    """
    return create_multi_period_scatter_chart(
        title="Momentum Quality vs Sharpe Ratio",
        x_title="Momentum Quality (R × Annualized Slope)",
        y_title="Sharpe Ratio (AR / SD)",
        x_data_func=lambda p: calculate_momentum_quality(momentum, p),
        y_data_func=lambda p: calculate_sharpe_ratio(performance, p),
        hover_template_func=lambda p: f'<b>%{{text}}</b><br>{p}M<br>Quality: %{{x:.3f}}<br>Sharpe: %{{y:.3f}}<extra></extra>',
        labels=momentum.index,
        add_horizontal_ref=False,
        add_vertical_ref=False
    )


def create_momentum_quality_vs_sortino_chart(momentum, performance):
    """
    Momentum Quality vs Sortino Ratio 차트를 생성합니다 (R × AS vs AR/DD).

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 데이터
    performance : pd.DataFrame
        Performance 데이터

    Returns:
    --------
    go.Figure
        생성된 차트
    """
    return create_multi_period_scatter_chart(
        title="Momentum Quality vs Sortino Ratio",
        x_title="Momentum Quality (R × Annualized Slope)",
        y_title="Sortino Ratio (AR / DD)",
        x_data_func=lambda p: calculate_momentum_quality(momentum, p),
        y_data_func=lambda p: calculate_sortino_ratio(performance, p),
        hover_template_func=lambda p: f'<b>%{{text}}</b><br>{p}M<br>Quality: %{{x:.3f}}<br>Sortino: %{{y:.3f}}<extra></extra>',
        labels=momentum.index,
        add_horizontal_ref=False,
        add_vertical_ref=False
    )


# ============================================================
# Dashboard Creation: Momentum
# ============================================================

def create_momentum_dashboard(momentum, performance):
    """
    Momentum 대시보드를 생성하고 저장합니다.

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 데이터
    performance : pd.DataFrame
        Performance 데이터
    """
    print("\n[1/4] Momentum Dashboard 생성 중...")

    # 출력 디렉토리 생성
    dashboard_dir = settings.output.dashboard_dir
    Path(dashboard_dir).mkdir(parents=True, exist_ok=True)

    # 차트 생성 (총 5개)
    chart_monthly = create_monthly_momentum_chart(momentum)
    chart_regression = create_regression_momentum_chart(momentum)
    chart_quality = create_momentum_quality_vs_return_chart(momentum, performance)
    chart_strength = create_momentum_strength_vs_return_chart(momentum, performance)
    chart_reliability = create_momentum_reliability_vs_return_chart(momentum, performance)

    # HTML 저장
    html_path = f'{dashboard_dir}/momentum.html'
    render_dashboard_html(
        title="KRX300 Momentum Analysis",
        figures=[chart_monthly, chart_regression, chart_quality, chart_strength, chart_reliability],
        chart_ids=['chart1', 'chart2', 'chart3', 'chart4', 'chart5'],
        output_path=html_path
    )
    print(f"  ✓ {html_path}")


# ============================================================
# Dashboard Creation: Performance
# ============================================================

def create_performance_dashboard(momentum, performance):
    """
    Performance 대시보드를 생성하고 저장합니다.

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 데이터
    performance : pd.DataFrame
        Performance 데이터
    """
    print("\n[2/4] Performance Dashboard 생성 중...")

    # 출력 디렉토리 생성
    dashboard_dir = settings.output.dashboard_dir
    Path(dashboard_dir).mkdir(parents=True, exist_ok=True)

    # 차트 생성 (총 4개)
    chart_sharpe = create_sharpe_ratio_chart(performance)
    chart_sortino = create_sortino_ratio_chart(performance)
    chart_quality_sharpe = create_momentum_quality_vs_sharpe_chart(momentum, performance)
    chart_quality_sortino = create_momentum_quality_vs_sortino_chart(momentum, performance)

    # HTML 저장
    html_path = f'{dashboard_dir}/performance.html'
    render_dashboard_html(
        title="KRX300 Performance Analysis",
        figures=[chart_sharpe, chart_sortino, chart_quality_sharpe, chart_quality_sortino],
        chart_ids=['chart1', 'chart2', 'chart3', 'chart4'],
        output_path=html_path
    )
    print(f"  ✓ {html_path}")


# ============================================================
# Dashboard Creation: Correlation Network
# ============================================================

def build_correlation_graph(corr_matrix, threshold):
    """
    상관관계 행렬로부터 NetworkX 그래프를 생성합니다.

    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        상관관계 행렬
    threshold : float
        엣지 생성 임계값

    Returns:
    --------
    nx.Graph
        생성된 그래프
    """
    tickers = corr_matrix.index.tolist()
    graph = nx.Graph()

    for i, ticker1 in enumerate(tickers):
        graph.add_node(ticker1)
        for j, ticker2 in enumerate(tickers):
            if i < j:
                corr_value = corr_matrix.loc[ticker1, ticker2]
                if corr_value > threshold:
                    graph.add_edge(ticker1, ticker2, weight=corr_value)

    return graph


def detect_communities(graph):
    """
    그래프에서 커뮤니티를 탐지합니다.

    Parameters:
    -----------
    graph : nx.Graph
        NetworkX 그래프

    Returns:
    --------
    dict
        노드 → 클러스터 ID 매핑
    """
    communities = nx.community.greedy_modularity_communities(graph)
    node_to_cluster = {}
    for cluster_id, community in enumerate(communities, 1):
        for node in community:
            node_to_cluster[node] = cluster_id
    return node_to_cluster


def create_vosviewer_json(graph, node_to_cluster, marginal_means, threshold):
    """
    VOSviewer 포맷 JSON을 생성합니다.

    Parameters:
    -----------
    graph : nx.Graph
        NetworkX 그래프
    node_to_cluster : dict
        노드 → 클러스터 ID 매핑
    marginal_means : pd.Series
        Marginal mean 값
    threshold : float
        상관관계 임계값

    Returns:
    --------
    dict
        VOSviewer JSON 데이터
    """
    # 레이아웃 설정
    layout_seed = settings.visualization.correlation_network.layout_seed
    layout_k = settings.visualization.correlation_network.layout_k
    pos = nx.spring_layout(graph, seed=layout_seed, k=layout_k)

    # 커뮤니티 수
    n_clusters = len(set(node_to_cluster.values()))

    # JSON 구조 생성
    vos_data = {
        "network": {
            "items": [],
            "links": [],
            "clusters": []
        },
        "config": {
            "parameters": {
                "min_cluster_size": 1
            }
        },
        "info": {
            "title": "KRX300 Correlation Network",
            "description": f"Correlation threshold: {threshold}"
        }
    }

    # 노드 추가
    for ticker in graph.nodes():
        vos_data["network"]["items"].append({
            "id": ticker,
            "label": ticker,
            "x": float(pos[ticker][0]),
            "y": float(pos[ticker][1]),
            "cluster": node_to_cluster.get(ticker, 1),
            "weights": {
                "correlation_sum": float(marginal_means[ticker])
            }
        })

    # 엣지 추가
    for edge in graph.edges(data=True):
        vos_data["network"]["links"].append({
            "source_id": edge[0],
            "target_id": edge[1],
            "strength": float(edge[2]['weight'])
        })

    # 클러스터 추가
    for cluster_id in range(1, n_clusters + 1):
        vos_data["network"]["clusters"].append({
            "cluster": cluster_id,
            "label": f"Cluster {cluster_id}"
        })

    return vos_data


def create_correlation_network(correlation):
    """
    Correlation Network를 생성하고 VOSviewer JSON으로 저장합니다.

    Parameters:
    -----------
    correlation : pd.DataFrame
        상관관계 행렬 (marginal mean 포함)
    """
    print("\n[3/4] Correlation Network 생성 중 (VOSviewer JSON)...")

    # 설정 로드
    threshold = settings.visualization.correlation_network.threshold
    dashboard_dir = settings.output.dashboard_dir
    Path(dashboard_dir).mkdir(parents=True, exist_ok=True)

    # marginal_mean 제거하고 실제 종목들만
    corr_matrix = correlation.drop(MARGINAL_MEAN_COLUMN, axis=0).drop(MARGINAL_MEAN_COLUMN, axis=1)
    tickers = corr_matrix.index.tolist()

    # 그래프 생성
    graph = build_correlation_graph(corr_matrix, threshold)
    print(f"  노드: {graph.number_of_nodes()}, 엣지: {graph.number_of_edges()}")

    # 고립 노드 확인
    isolated_nodes = [node for node in graph.nodes() if graph.degree(node) == 0]
    if isolated_nodes:
        print(f"  경고: {len(isolated_nodes)}개 고립 노드 발견 (threshold={threshold})")

    # 커뮤니티 탐지
    node_to_cluster = detect_communities(graph)
    n_clusters = len(set(node_to_cluster.values()))
    print(f"  클러스터: {n_clusters}개")

    # VOSviewer JSON 생성
    marginal_means = correlation.loc[tickers, MARGINAL_MEAN_COLUMN]
    vos_data = create_vosviewer_json(graph, node_to_cluster, marginal_means, threshold)

    # JSON 저장
    json_path = f'{dashboard_dir}/correlation_network.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(vos_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {json_path}")

    # VOSviewer 안내 HTML 생성
    html_path = f'{dashboard_dir}/correlation_network.html'
    render_data = {
        'title': 'KRX300 Correlation Network - VOSviewer',
        'n_nodes': graph.number_of_nodes(),
        'n_edges': graph.number_of_edges(),
        'n_clusters': n_clusters,
        'threshold': threshold,
        'n_isolated': len(isolated_nodes)
    }
    render_html_from_template('correlation_network.html', render_data, html_path)
    print(f"  ✓ {html_path}")


# ============================================================
# Dashboard Creation: Correlation Cluster
# ============================================================

def perform_hierarchical_clustering(corr_matrix, n_clusters, method):
    """
    계층적 클러스터링을 수행합니다.

    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        상관관계 행렬
    n_clusters : int
        클러스터 개수
    method : str
        클러스터링 방법 (ward, average, complete 등)

    Returns:
    --------
    tuple
        (linkage_matrix, cluster_labels, clusters_dict)
    """
    tickers = corr_matrix.index.tolist()

    # 상관계수를 거리로 변환 (1 - correlation)
    distance_matrix = 1 - corr_matrix

    # 거리 행렬을 condensed form으로 변환
    condensed_dist = squareform(distance_matrix, checks=False)

    # Hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method=method)

    # Cluster assignments
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    # 클러스터별 ticker 추출
    clusters = {}
    for i in range(1, n_clusters + 1):
        cluster_tickers = [tickers[j] for j in range(len(tickers)) if cluster_labels[j] == i]
        if cluster_tickers:
            clusters[f"Cluster_{i}"] = cluster_tickers

    return linkage_matrix, cluster_labels, clusters


def create_dendrogram_figure(distance_matrix, tickers, method):
    """
    덴드로그램 Figure를 생성합니다.

    Parameters:
    -----------
    distance_matrix : pd.DataFrame
        거리 행렬
    tickers : list
        종목 코드 리스트
    method : str
        클러스터링 방법

    Returns:
    --------
    go.Figure
        생성된 덴드로그램
    """
    width = settings.dashboard.width
    height_per_item = settings.visualization.dendrogram.height_per_item

    fig = ff.create_dendrogram(
        distance_matrix.values,
        orientation='left',
        labels=tickers,
        linkagefun=lambda x: linkage(x, method=method)
    )

    height = max(800, len(tickers) * height_per_item)

    fig.update_layout(
        title="KRX300 Hierarchical Clustering (Correlation-based)",
        width=width,
        height=height,
        xaxis_title="Distance (1 - Correlation)",
        yaxis_title="Stocks",
        hovermode='closest'
    )

    return fig


def create_correlation_cluster(correlation):
    """
    Correlation Cluster Dendrogram을 생성하고 저장합니다.

    Parameters:
    -----------
    correlation : pd.DataFrame
        상관관계 행렬 (marginal mean 포함)
    """
    print("\n[4/4] Correlation Cluster Dendrogram 생성 중...")

    # 설정 로드
    cluster_method = settings.visualization.dendrogram.method
    n_clusters = settings.visualization.dendrogram.n_cluster
    dashboard_dir = settings.output.dashboard_dir
    Path(dashboard_dir).mkdir(parents=True, exist_ok=True)

    # marginal_mean 제거
    corr_matrix = correlation.drop(MARGINAL_MEAN_COLUMN, axis=0).drop(MARGINAL_MEAN_COLUMN, axis=1)
    tickers = corr_matrix.index.tolist()

    # 계층적 클러스터링
    linkage_matrix, cluster_labels, clusters = perform_hierarchical_clustering(
        corr_matrix, n_clusters, cluster_method
    )

    print(f"  클러스터 수: {len(clusters)}개")
    print(f"  평균 클러스터 크기: {len(tickers) / len(clusters):.1f}개")

    # 덴드로그램 생성
    distance_matrix = 1 - corr_matrix
    fig = create_dendrogram_figure(distance_matrix, tickers, cluster_method)

    # 클러스터 정보 JSON 저장
    cluster_data = {
        'metadata': {
            'n_clusters': len(clusters),
            'n_tickers': len(tickers),
            'method': cluster_method,
            'avg_cluster_size': len(tickers) / len(clusters)
        },
        'clusters': clusters
    }

    json_path = f'{dashboard_dir}/correlation_cluster.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {json_path}")

    # HTML 생성
    html_path = f'{dashboard_dir}/correlation_cluster.html'
    render_html_from_template(
        'correlation_cluster.html',
        {
            'title': 'KRX300 Correlation Cluster',
            'dendrogram_html': fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                div_id='dendrogram',
                config={'responsive': True}
            ),
            'n_clusters': len(clusters),
            'n_tickers': len(tickers),
            'method': cluster_method,
            'avg_cluster_size': f"{len(tickers) / len(clusters):.1f}",
            'clusters': [
                {
                    'name': name,
                    'size': len(ticker_list),
                    'tickers': ', '.join(sorted(ticker_list))
                }
                for name, ticker_list in sorted(clusters.items())
            ]
        },
        html_path
    )
    print(f"  ✓ {html_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("STEP 4: KRX300 대시보드 생성")
    print("=" * 70)

    # 데이터 로드
    print("\n데이터 로딩 중...")
    signal_dir = settings.output.signal_dir

    momentum = import_dataframe_from_json(f'{signal_dir}/momentum.json')
    performance = import_dataframe_from_json(f'{signal_dir}/performance.json')
    correlation = import_dataframe_from_json(f'{signal_dir}/correlation.json')

    print(f"  Momentum: {momentum.shape}")
    print(f"  Performance: {performance.shape}")
    print(f"  Correlation: {correlation.shape}")

    # 대시보드 생성
    create_momentum_dashboard(momentum, performance)
    create_performance_dashboard(momentum, performance)
    create_correlation_network(correlation)
    create_correlation_cluster(correlation)

    print("\n" + "=" * 70)
    print("STEP 4 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
