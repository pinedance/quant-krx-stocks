"""
시각화 모듈

Plotly 기반 인터랙티브 차트와 네트워크 분석 시각화를 제공합니다.
- 레이어 아키텍처 (Layer 0: 헬퍼, Layer 1: 공개 API)
- 재사용 가능한 빌더 패턴
- NetworkX 기반 상관관계 네트워크 분석

구성:
-----
- Layer 0: Plotly 차트 헬퍼 함수 (private)
- Layer 1: 공개 차트 생성 함수
  - Momentum 차트 (6개)
  - Performance 차트 (4개)
  - 네트워크/클러스터 분석 (4개)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from core.config import settings
from core.finance import (
    calculate_momentum_quality,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_correlation_coefficient
)


# ============================================================
# 내부 상수 (Minor Constants)
# ============================================================

_MIN_TRENDLINE_POINTS = 2  # 추세선 생성 최소 데이터 포인트
_MIN_DENDROGRAM_HEIGHT = 800  # 덴드로그램 최소 높이
_CLUSTER_ID_START = 1  # 클러스터 ID 시작 번호


# ============================================================
# Layer 0: 내부 헬퍼 함수 (Private)
# ============================================================

def _calculate_axis_range(data_series: pd.Series) -> list:
    """
    축 범위를 계산합니다 (순수 함수).

    데이터의 최소/최대값에 패딩을 적용하여 차트 축 범위를 계산합니다.

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


def _add_trendline(
    fig: go.Figure,
    x_data: pd.Series,
    y_data: pd.Series,
    color: str,
    name: str = None,
    show_in_legend: bool = True,
    legendgroup: str = None
):
    """
    산점도에 추세선을 추가합니다 (순수 함수).

    선형 회귀(1차 다항식)를 사용하여 추세선을 계산하고 Figure에 추가합니다.

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
    if non_null_mask.sum() <= _MIN_TRENDLINE_POINTS:
        return

    x_valid = x_data[non_null_mask]
    y_valid = y_data[non_null_mask]

    # 선형 회귀 (1차 다항식)
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


def _add_scatter_trace(
    fig: go.Figure,
    x_data: pd.Series,
    y_data: pd.Series,
    labels: pd.Index,
    color: str,
    name: str,
    hover_template: str,
    legendgroup: str = None
):
    """
    산점도 trace를 추가합니다 (순수 함수).

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


def _add_reference_lines(
    fig: go.Figure,
    add_horizontal: bool = True,
    add_vertical: bool = False
):
    """
    참조선(0을 나타내는 선)을 추가합니다 (순수 함수).

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


def _create_base_figure(
    title: str,
    x_title: str,
    y_title: str,
    x_range: list,
    y_range: list
) -> go.Figure:
    """
    기본 Figure 객체를 생성하고 레이아웃을 설정합니다 (순수 함수).

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
# Layer 1: 공개 API (Public Interface)
# ============================================================

# ------------------------------------------------------------
# 복합 차트 빌더
# ------------------------------------------------------------

def create_multi_period_scatter_chart(
    title: str,
    x_title: str,
    y_title: str,
    x_data_func,
    y_data_func,
    hover_template_func,
    labels: pd.Index,
    periods: list = None,
    colors: list = None,
    add_horizontal_ref: bool = True,
    add_vertical_ref: bool = False
) -> go.Figure:
    """
    여러 기간 데이터를 표시하는 산점도 차트를 생성합니다.

    이 함수는 다양한 기간별 차트를 생성하기 위한 재사용 가능한 빌더입니다.

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

    x_range = _calculate_axis_range(pd.concat(all_x_data))
    y_range = _calculate_axis_range(pd.concat(all_y_data))

    # Figure 생성
    fig = _create_base_figure(title, x_title, y_title, x_range, y_range)

    # 각 기간별 trace 추가
    for period, color in zip(periods, colors):
        x_data = x_data_func(period)
        y_data = y_data_func(period)

        # 같은 legendgroup으로 scatter와 trendline 묶기
        group_name = f'period_{period}'

        _add_scatter_trace(
            fig=fig,
            x_data=x_data,
            y_data=y_data,
            labels=labels,
            color=color,
            name=f'{period}개월',
            hover_template=hover_template_func(period),
            legendgroup=group_name
        )

        _add_trendline(
            fig=fig,
            x_data=x_data,
            y_data=y_data,
            color=color,
            name=f'{period}M 추세선',
            show_in_legend=False,
            legendgroup=group_name
        )

    # 참조선
    _add_reference_lines(fig, add_horizontal_ref, add_vertical_ref)

    return fig


# ------------------------------------------------------------
# 개별 차트 생성: Momentum
# ------------------------------------------------------------

def create_monthly_momentum_chart(momentum: pd.DataFrame) -> go.Figure:
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


def create_regression_momentum_chart(momentum: pd.DataFrame) -> go.Figure:
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


def create_momentum_quality_vs_return_chart(
    momentum: pd.DataFrame,
    performance: pd.DataFrame
) -> go.Figure:
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


def create_momentum_strength_vs_return_chart(
    momentum: pd.DataFrame,
    performance: pd.DataFrame
) -> go.Figure:
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


def create_momentum_reliability_vs_return_chart(
    momentum: pd.DataFrame,
    performance: pd.DataFrame
) -> go.Figure:
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


# ------------------------------------------------------------
# 개별 차트 생성: Performance
# ------------------------------------------------------------

def create_sharpe_ratio_chart(performance: pd.DataFrame) -> go.Figure:
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


def create_sortino_ratio_chart(performance: pd.DataFrame) -> go.Figure:
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


def create_momentum_quality_vs_sharpe_chart(
    momentum: pd.DataFrame,
    performance: pd.DataFrame
) -> go.Figure:
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


def create_momentum_quality_vs_sortino_chart(
    momentum: pd.DataFrame,
    performance: pd.DataFrame
) -> go.Figure:
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


# ------------------------------------------------------------
# 네트워크/클러스터 분석
# ------------------------------------------------------------

def create_correlation_graph(
    corr_matrix: pd.DataFrame,
    threshold: float
) -> nx.Graph:
    """
    상관관계 행렬로부터 NetworkX 그래프를 생성합니다.

    임계값보다 높은 상관계수를 가진 종목 쌍 사이에 엣지를 생성합니다.

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


def create_communities(graph: nx.Graph) -> dict:
    """
    그래프에서 커뮤니티를 탐지합니다.

    Greedy modularity maximization 알고리즘을 사용합니다.

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

    for cluster_id, community in enumerate(communities, _CLUSTER_ID_START):
        for node in community:
            node_to_cluster[node] = cluster_id

    return node_to_cluster


def create_vosviewer_json(
    graph: nx.Graph,
    node_to_cluster: dict,
    marginal_means: pd.Series,
    threshold: float
) -> dict:
    """
    VOSviewer 포맷 JSON을 생성합니다.

    VOSviewer는 네트워크 시각화 도구로, 이 함수는 해당 형식의 JSON을 생성합니다.

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
            "title": "KRX Correlation Network",
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
            "cluster": node_to_cluster.get(ticker, _CLUSTER_ID_START),
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
    for cluster_id in range(_CLUSTER_ID_START, n_clusters + _CLUSTER_ID_START):
        vos_data["network"]["clusters"].append({
            "cluster": cluster_id,
            "label": f"Cluster {cluster_id}"
        })

    return vos_data


def create_hierarchical_clusters(
    corr_matrix: pd.DataFrame,
    n_clusters: int,
    method: str
) -> tuple:
    """
    계층적 클러스터링을 수행합니다.

    상관계수를 거리로 변환(1 - correlation)하여 계층적 클러스터링을 수행합니다.

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
        - linkage_matrix: 계층적 클러스터링 결과
        - cluster_labels: 각 종목의 클러스터 ID
        - clusters_dict: 클러스터별 종목 리스트
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
    for i in range(_CLUSTER_ID_START, n_clusters + _CLUSTER_ID_START):
        cluster_tickers = [
            tickers[j] for j in range(len(tickers))
            if cluster_labels[j] == i
        ]
        if cluster_tickers:
            clusters[f"Cluster_{i}"] = cluster_tickers

    return linkage_matrix, cluster_labels, clusters


def create_dendrogram_figure(
    distance_matrix: pd.DataFrame,
    tickers: list,
    method: str
) -> go.Figure:
    """
    덴드로그램 Figure를 생성합니다.

    계층적 클러스터링 결과를 트리 형태로 시각화합니다.

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

    height = max(_MIN_DENDROGRAM_HEIGHT, len(tickers) * height_per_item)

    fig.update_layout(
        title="KRX Hierarchical Clustering (Correlation-based)",
        width=width,
        height=height,
        xaxis_title="Distance (1 - Correlation)",
        yaxis_title="Stocks",
        hovermode='closest'
    )

    return fig
