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
# Chart Creation: Momentum
# ============================================================

def create_monthly_momentum_chart(momentum):
    """
    Monthly Momentum 차트를 생성합니다 (1MR vs 13612MR).

    Parameters:
    -----------
    momentum : pd.DataFrame
        Momentum 데이터

    Returns:
    --------
    go.Figure
        생성된 차트
    """
    x_data = momentum['1MR']
    y_data = momentum['13612MR']

    # 축 범위 계산
    x_range = calculate_axis_range(x_data)
    y_range = calculate_axis_range(y_data)

    # Figure 생성
    fig = create_base_figure(
        title="Monthly Momentum",
        x_title="1개월 모멘텀 (1MR)",
        y_title="평균 모멘텀 (13612MR)",
        x_range=x_range,
        y_range=y_range
    )

    # 산점도 추가
    add_scatter_trace(
        fig=fig,
        x_data=x_data,
        y_data=y_data,
        labels=momentum.index,
        color='royalblue',
        name='Stocks',
        hover_template='<b>%{text}</b><br>1MR: %{x:.2%}<br>13612MR: %{y:.2%}<extra></extra>'
    )

    # 추세선 추가
    trendline_color = settings.visualization.scatter_plot.trendline.single_color
    add_trendline(fig, x_data, y_data, color=trendline_color)

    # 참조선 추가
    add_reference_lines(fig, add_horizontal=True, add_vertical=True)

    return fig


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
    periods = settings.visualization.scatter_plot.periods
    colors = settings.visualization.scatter_plot.colors

    # 모든 데이터를 수집하여 축 범위 계산
    all_x_data = []
    all_y_data = []
    for period in periods:
        all_x_data.append(momentum[f'RS{period}'])
        all_y_data.append(momentum[f'AS{period}'])

    x_range = calculate_axis_range(pd.concat(all_x_data))
    y_range = calculate_axis_range(pd.concat(all_y_data))

    # Figure 생성
    fig = create_base_figure(
        title="Regression Momentum",
        x_title="R-squared",
        y_title="Annualized Slope",
        x_range=x_range,
        y_range=y_range
    )

    # 각 기간별 산점도 및 추세선 추가
    for period, color in zip(periods, colors):
        x_data = momentum[f'RS{period}']
        y_data = momentum[f'AS{period}']

        # 산점도
        add_scatter_trace(
            fig=fig,
            x_data=x_data,
            y_data=y_data,
            labels=momentum.index,
            color=color,
            name=f'{period}개월',
            hover_template=f'<b>%{{text}}</b><br>{period}M<br>R²: %{{x:.3f}}<br>Ann. Slope: %{{y:.2%}}<extra></extra>'
        )

        # 추세선
        add_trendline(
            fig=fig,
            x_data=x_data,
            y_data=y_data,
            color=color,
            name=f'{period}M 추세선',
            show_in_legend=False
        )

    # 참조선 추가
    add_reference_lines(fig, add_horizontal=True, add_vertical=False)

    return fig


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
    periods = settings.visualization.scatter_plot.periods
    colors = settings.visualization.scatter_plot.colors

    # 모든 데이터를 수집하여 축 범위 계산
    all_x_data = []
    all_y_data = []
    for period in periods:
        # √R² × Annualized Slope = R × Annualized Slope
        momentum_quality = np.sqrt(momentum[f'RS{period}']) * momentum[f'AS{period}']
        all_x_data.append(momentum_quality)
        all_y_data.append(performance[f'AR{period}'])

    x_range = calculate_axis_range(pd.concat(all_x_data))
    y_range = calculate_axis_range(pd.concat(all_y_data))

    # Figure 생성
    fig = create_base_figure(
        title="Momentum Quality vs Return",
        x_title="Momentum Quality (R × Annualized Slope)",
        y_title="Annualized Return",
        x_range=x_range,
        y_range=y_range
    )

    # 각 기간별 산점도 및 추세선 추가
    for period, color in zip(periods, colors):
        x_data = np.sqrt(momentum[f'RS{period}']) * momentum[f'AS{period}']
        y_data = performance[f'AR{period}']

        # 산점도
        add_scatter_trace(
            fig=fig,
            x_data=x_data,
            y_data=y_data,
            labels=momentum.index,
            color=color,
            name=f'{period}개월',
            hover_template=f'<b>%{{text}}</b><br>{period}M<br>Quality: %{{x:.3f}}<br>AR: %{{y:.2%}}<extra></extra>'
        )

        # 추세선
        add_trendline(
            fig=fig,
            x_data=x_data,
            y_data=y_data,
            color=color,
            name=f'{period}M 추세선',
            show_in_legend=False
        )

    # 참조선 추가
    add_reference_lines(fig, add_horizontal=True, add_vertical=False)

    return fig


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
    periods = settings.visualization.scatter_plot.periods
    colors = settings.visualization.scatter_plot.colors

    # 모든 데이터를 수집하여 축 범위 계산
    all_x_data = []
    all_y_data = []
    for period in periods:
        all_x_data.append(momentum[f'AS{period}'])
        all_y_data.append(performance[f'AR{period}'])

    x_range = calculate_axis_range(pd.concat(all_x_data))
    y_range = calculate_axis_range(pd.concat(all_y_data))

    # Figure 생성
    fig = create_base_figure(
        title="Momentum Strength vs Return",
        x_title="Annualized Slope",
        y_title="Annualized Return",
        x_range=x_range,
        y_range=y_range
    )

    # 각 기간별 산점도 및 추세선 추가
    for period, color in zip(periods, colors):
        x_data = momentum[f'AS{period}']
        y_data = performance[f'AR{period}']

        # 산점도
        add_scatter_trace(
            fig=fig,
            x_data=x_data,
            y_data=y_data,
            labels=momentum.index,
            color=color,
            name=f'{period}개월',
            hover_template=f'<b>%{{text}}</b><br>{period}M<br>AS: %{{x:.2%}}<br>AR: %{{y:.2%}}<extra></extra>'
        )

        # 추세선
        add_trendline(
            fig=fig,
            x_data=x_data,
            y_data=y_data,
            color=color,
            name=f'{period}M 추세선',
            show_in_legend=False
        )

    # 참조선 추가
    add_reference_lines(fig, add_horizontal=True, add_vertical=False)

    return fig


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
    periods = settings.visualization.scatter_plot.periods
    colors = settings.visualization.scatter_plot.colors

    # 모든 데이터를 수집하여 축 범위 계산
    all_x_data = []
    all_y_data = []
    for period in periods:
        # √R² = R (상관계수)
        all_x_data.append(np.sqrt(momentum[f'RS{period}']))
        all_y_data.append(performance[f'AR{period}'])

    x_range = calculate_axis_range(pd.concat(all_x_data))
    y_range = calculate_axis_range(pd.concat(all_y_data))

    # Figure 생성
    fig = create_base_figure(
        title="Momentum Reliability vs Return",
        x_title="Correlation Coefficient (R)",
        y_title="Annualized Return",
        x_range=x_range,
        y_range=y_range
    )

    # 각 기간별 산점도 및 추세선 추가
    for period, color in zip(periods, colors):
        x_data = np.sqrt(momentum[f'RS{period}'])
        y_data = performance[f'AR{period}']

        # 산점도
        add_scatter_trace(
            fig=fig,
            x_data=x_data,
            y_data=y_data,
            labels=momentum.index,
            color=color,
            name=f'{period}개월',
            hover_template=f'<b>%{{text}}</b><br>{period}M<br>R: %{{x:.3f}}<br>AR: %{{y:.2%}}<extra></extra>'
        )

        # 추세선
        add_trendline(
            fig=fig,
            x_data=x_data,
            y_data=y_data,
            color=color,
            name=f'{period}M 추세선',
            show_in_legend=False
        )

    # 참조선 추가
    add_reference_lines(fig, add_horizontal=True, add_vertical=False)

    return fig


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
    periods = settings.visualization.scatter_plot.periods
    colors = settings.visualization.scatter_plot.colors

    # 모든 데이터를 수집하여 축 범위 계산
    all_x_data = []
    all_y_data = []
    for period in periods:
        all_x_data.append(performance[f'SD{period}'])
        all_y_data.append(performance[f'AR{period}'])

    x_range = calculate_axis_range(pd.concat(all_x_data))
    y_range = calculate_axis_range(pd.concat(all_y_data))

    # Figure 생성
    fig = create_base_figure(
        title="Sharpe Ratio",
        x_title="Standard Deviation",
        y_title="Annualized Return",
        x_range=x_range,
        y_range=y_range
    )

    # 각 기간별 산점도 및 추세선 추가
    for period, color in zip(periods, colors):
        x_data = performance[f'SD{period}']
        y_data = performance[f'AR{period}']

        # 산점도
        add_scatter_trace(
            fig=fig,
            x_data=x_data,
            y_data=y_data,
            labels=performance.index,
            color=color,
            name=f'{period}개월',
            hover_template=f'<b>%{{text}}</b><br>{period}M<br>SD: %{{x:.2%}}<br>AR: %{{y:.2%}}<extra></extra>'
        )

        # 추세선
        add_trendline(
            fig=fig,
            x_data=x_data,
            y_data=y_data,
            color=color,
            name=f'{period}M 추세선',
            show_in_legend=False
        )

    return fig


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
    periods = settings.visualization.scatter_plot.periods
    colors = settings.visualization.scatter_plot.colors

    # 모든 데이터를 수집하여 축 범위 계산
    all_x_data = []
    all_y_data = []
    for period in periods:
        all_x_data.append(performance[f'DD{period}'])
        all_y_data.append(performance[f'AR{period}'])

    x_range = calculate_axis_range(pd.concat(all_x_data))
    y_range = calculate_axis_range(pd.concat(all_y_data))

    # Figure 생성
    fig = create_base_figure(
        title="Sortino Ratio",
        x_title="Downside Deviation",
        y_title="Annualized Return",
        x_range=x_range,
        y_range=y_range
    )

    # 각 기간별 산점도 및 추세선 추가
    for period, color in zip(periods, colors):
        x_data = performance[f'DD{period}']
        y_data = performance[f'AR{period}']

        # 산점도
        add_scatter_trace(
            fig=fig,
            x_data=x_data,
            y_data=y_data,
            labels=performance.index,
            color=color,
            name=f'{period}개월',
            hover_template=f'<b>%{{text}}</b><br>{period}M<br>DD: %{{x:.2%}}<br>AR: %{{y:.2%}}<extra></extra>'
        )

        # 추세선
        add_trendline(
            fig=fig,
            x_data=x_data,
            y_data=y_data,
            color=color,
            name=f'{period}M 추세선',
            show_in_legend=False
        )

    return fig


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

def create_performance_dashboard(performance):
    """
    Performance 대시보드를 생성하고 저장합니다.

    Parameters:
    -----------
    performance : pd.DataFrame
        Performance 데이터
    """
    print("\n[2/4] Performance Dashboard 생성 중...")

    # 출력 디렉토리 생성
    dashboard_dir = settings.output.dashboard_dir
    Path(dashboard_dir).mkdir(parents=True, exist_ok=True)

    # 차트 생성
    chart_sharpe = create_sharpe_ratio_chart(performance)
    chart_sortino = create_sortino_ratio_chart(performance)

    # HTML 저장
    html_path = f'{dashboard_dir}/performance.html'
    render_dashboard_html(
        title="KRX300 Performance Analysis",
        figures=[chart_sharpe, chart_sortino],
        chart_ids=['chart1', 'chart2'],
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
    corr_matrix = correlation.drop('mean', axis=0).drop('mean', axis=1)
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
    marginal_means = correlation.loc[tickers, 'mean']
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
    corr_matrix = correlation.drop('mean', axis=0).drop('mean', axis=1)
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
    create_performance_dashboard(performance)
    create_correlation_network(correlation)
    create_correlation_cluster(correlation)

    print("\n" + "=" * 70)
    print("STEP 4 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
