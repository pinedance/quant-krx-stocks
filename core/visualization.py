"""
시각화 모듈

이 모듈은 Plotly 기반 인터랙티브 차트 생성을 위한 함수들을 제공합니다.

구성:
-----
- 지표 계산 함수 (momentum quality, sharpe ratio 등)
- Plotly 차트 헬퍼 함수 (axis, trendline, scatter 등)
- 복합 차트 빌더 (multi-period scatter 등)
- 개별 차트 생성 함수 (momentum, performance 차트)
- 네트워크/클러스터 분석 및 시각화
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from core.config import settings


# ============================================================
# 지표 계산 함수
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
# Plotly 차트 헬퍼 함수
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


def add_trendline(fig, x_data, y_data, color, name=None, show_in_legend=True, legendgroup=None):
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
    legendgroup : str, optional
        Legend 그룹 이름 (같은 그룹은 함께 show/hide됨)
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
            legendgroup=legendgroup,
            hoverinfo='skip'
        )
    )


def add_scatter_trace(fig, x_data, y_data, labels, color, name, hover_template, legendgroup=None):
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
    legendgroup : str, optional
        Legend 그룹 이름 (같은 그룹은 함께 show/hide됨)
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
            name=name,
            legendgroup=legendgroup
        )
    )


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
    scatter_config = settings.visualization.scatter_plot
    layout_style = scatter_config.layout
    height = scatter_config.height
    width = scatter_config.width

    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis_range=x_range,
        yaxis_range=y_range,
        height=height,
        width=width,
        showlegend=layout_style.showlegend,
        hovermode=layout_style.hovermode
    )

    return fig


# ============================================================
# 복합 차트 빌더
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

        # 같은 legendgroup으로 scatter와 trendline 묶기
        group_name = f'period_{period}'

        add_scatter_trace(
            fig=fig,
            x_data=x_data,
            y_data=y_data,
            labels=labels,
            color=color,
            name=f'{period}개월',
            hover_template=hover_template_func(period),
            legendgroup=group_name
        )

        add_trendline(
            fig=fig,
            x_data=x_data,
            y_data=y_data,
            color=color,
            name=f'{period}M 추세선',
            show_in_legend=False,
            legendgroup=group_name
        )

    # 참조선
    add_reference_lines(fig, add_horizontal_ref, add_vertical_ref)

    return fig


# ============================================================
# 개별 차트 생성: Momentum
# ============================================================

def create_monthly_momentum_chart(momentum):
    """
    Correlation vs Average Momentum 차트를 생성합니다 (R vs 13612MR, 12개월 기준).

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
        title="Correlation vs Average Momentum",
        x_title="Correlation Coefficient (R)",
        y_title="Average Momentum (13612MR)",
        x_data_func=lambda p: calculate_correlation_coefficient(momentum, p),
        y_data_func=lambda p: momentum['13612MR'],
        hover_template_func=lambda p: '<b>%{text}</b><br>R: %{x:.3f}<br>Avg Momentum: %{y:.2%}<extra></extra>',
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
        x_title="R-squared (Reliability)",
        y_title="Annualized Slope (Strength)",
        x_data_func=lambda p: momentum[f'RS{p}'],
        y_data_func=lambda p: momentum[f'AS{p}'],
        hover_template_func=lambda p: f'<b>%{{text}}</b><br>{p}M<br>R²: %{{x:.3f}}<br>Slope: %{{y:.2%}}<extra></extra>',
        labels=momentum.index,
        add_horizontal_ref=True,
        add_vertical_ref=False
    )


def create_momentum_quality_vs_return_chart(momentum, performance):
    """
    Momentum Quality vs Return 차트를 생성합니다 (Momentum Quality vs AR).

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
        x_title="Momentum Quality (R × Slope)",
        y_title="Annualized Return",
        x_data_func=lambda p: calculate_momentum_quality(momentum, p),
        y_data_func=lambda p: performance[f'AR{p}'],
        hover_template_func=lambda p: f'<b>%{{text}}</b><br>{p}M<br>Quality: %{{x:.3f}}<br>Return: %{{y:.2%}}<extra></extra>',
        labels=momentum.index,
        add_horizontal_ref=True,
        add_vertical_ref=False
    )


def create_momentum_strength_vs_return_chart(momentum, performance):
    """
    Momentum Strength vs Return 차트를 생성합니다 (Annualized Slope vs AR).

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
        x_title="Momentum Strength (Annualized Slope)",
        y_title="Annualized Return",
        x_data_func=lambda p: momentum[f'AS{p}'],
        y_data_func=lambda p: performance[f'AR{p}'],
        hover_template_func=lambda p: f'<b>%{{text}}</b><br>{p}M<br>Slope: %{{x:.2%}}<br>Return: %{{y:.2%}}<extra></extra>',
        labels=momentum.index,
        add_horizontal_ref=True,
        add_vertical_ref=False
    )


def create_momentum_reliability_vs_return_chart(momentum, performance):
    """
    Momentum Reliability vs Return 차트를 생성합니다 (Correlation Coefficient vs AR).

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
        x_title="Momentum Reliability (Correlation R)",
        y_title="Annualized Return",
        x_data_func=lambda p: calculate_correlation_coefficient(momentum, p),
        y_data_func=lambda p: performance[f'AR{p}'],
        hover_template_func=lambda p: f'<b>%{{text}}</b><br>{p}M<br>R: %{{x:.3f}}<br>Return: %{{y:.2%}}<extra></extra>',
        labels=momentum.index,
        add_horizontal_ref=True,
        add_vertical_ref=False
    )


# ============================================================
# 개별 차트 생성: Performance
# ============================================================

def create_sharpe_ratio_chart(performance):
    """
    Risk-Return Analysis 차트를 생성합니다 (SD vs AR, Sharpe Ratio 구성 요소).

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
        title="Risk-Return Analysis (Sharpe Ratio)",
        x_title="Standard Deviation (Risk)",
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
    Downside Risk-Return Analysis 차트를 생성합니다 (DD vs AR, Sortino Ratio 구성 요소).

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
        title="Downside Risk-Return Analysis (Sortino Ratio)",
        x_title="Downside Deviation (Downside Risk)",
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
    Momentum Quality vs Sharpe Ratio 차트를 생성합니다 (R × AS vs Sharpe Ratio).

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
        x_title="Momentum Quality (R × Slope)",
        y_title="Sharpe Ratio (Return / Risk)",
        x_data_func=lambda p: calculate_momentum_quality(momentum, p),
        y_data_func=lambda p: calculate_sharpe_ratio(performance, p),
        hover_template_func=lambda p: f'<b>%{{text}}</b><br>{p}M<br>Quality: %{{x:.3f}}<br>Sharpe: %{{y:.3f}}<extra></extra>',
        labels=momentum.index,
        add_horizontal_ref=False,
        add_vertical_ref=False
    )


def create_momentum_quality_vs_sortino_chart(momentum, performance):
    """
    Momentum Quality vs Sortino Ratio 차트를 생성합니다 (R × AS vs Sortino Ratio).

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
        x_title="Momentum Quality (R × Slope)",
        y_title="Sortino Ratio (Return / Downside Risk)",
        x_data_func=lambda p: calculate_momentum_quality(momentum, p),
        y_data_func=lambda p: calculate_sortino_ratio(performance, p),
        hover_template_func=lambda p: f'<b>%{{text}}</b><br>{p}M<br>Quality: %{{x:.3f}}<br>Sortino: %{{y:.3f}}<extra></extra>',
        labels=momentum.index,
        add_horizontal_ref=False,
        add_vertical_ref=False
    )


# ============================================================
# 네트워크/클러스터 분석
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
    dendrogram_config = settings.visualization.dendrogram
    width = dendrogram_config.width
    height_per_item = dendrogram_config.height_per_item

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
