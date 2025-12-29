"""
STEP 4: KRX300 대시보드 생성
- Momentum Dashboard (Plotly)
- Performance Dashboard (Plotly)
- Correlation Network (VOSviewer JSON)
- Correlation Cluster (Plotly Dendrogram)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import networkx as nx
from core.config import settings
from core.io import import_dataframe_from_json, render_dashboard_html, render_html_from_template


def create_momentum_dashboard(momentum):
    """Momentum 대시보드 생성 (독립된 플롯 2개를 하나의 HTML에)"""
    print("\n[1/4] Momentum Dashboard 생성 중...")

    periods = settings.visualization.scatter_plot.periods
    colors = settings.visualization.scatter_plot.colors
    height = settings.dashboard.height
    dashboard_dir = settings.output.dashboard_dir
    Path(dashboard_dir).mkdir(parents=True, exist_ok=True)

    # ========== Chart 1: Monthly Momentum ==========
    x1 = momentum['1MR']
    y1 = momentum['13612MR']

    fig1 = go.Figure()

    # 데이터 포인트
    fig1.add_trace(
        go.Scatter(
            x=x1,
            y=y1,
            mode='markers',
            marker=dict(size=8, color='royalblue', opacity=0.6),
            text=momentum.index,
            hovertemplate='<b>%{text}</b><br>1MR: %{x:.2%}<br>13612MR: %{y:.2%}<extra></extra>',
            name='Stocks'
        )
    )

    # 추세선
    valid_mask = ~(x1.isna() | y1.isna())
    if valid_mask.sum() > 1:
        x1_valid = x1[valid_mask]
        y1_valid = y1[valid_mask]
        z1 = np.polyfit(x1_valid, y1_valid, 1)
        p1 = np.poly1d(z1)
        x1_trend = np.array([x1_valid.min(), x1_valid.max()])
        y1_trend = p1(x1_trend)

        fig1.add_trace(
            go.Scatter(
                x=x1_trend,
                y=y1_trend,
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name=f'추세선 (y={z1[0]:.2f}x+{z1[1]:.2f})',
                hoverinfo='skip'
            )
        )

    # 사분면 구분선
    fig1.add_hline(y=0, line_dash="dash", line_color="gray")
    fig1.add_vline(x=0, line_dash="dash", line_color="gray")

    # 축 범위
    x1_range = [x1.min() * 1.1 if x1.min() < 0 else x1.min() * 0.9,
                x1.max() * 1.1 if x1.max() > 0 else x1.max() * 0.9]
    y1_range = [y1.min() * 1.1 if y1.min() < 0 else y1.min() * 0.9,
                y1.max() * 1.1 if y1.max() > 0 else y1.max() * 0.9]

    fig1.update_layout(
        title="Monthly Momentum",
        xaxis_title="1개월 모멘텀 (1MR)",
        yaxis_title="평균 모멘텀 (13612MR)",
        xaxis_range=x1_range,
        yaxis_range=y1_range,
        height=height,
        showlegend=True,
        hovermode='closest'
    )

    # ========== Chart 2: Regression Momentum ==========
    fig2 = go.Figure()

    for period, color in zip(periods, colors):
        x2 = momentum[f'RS{period}']
        y2 = momentum[f'AS{period}']

        # 데이터 포인트
        fig2.add_trace(
            go.Scatter(
                x=x2,
                y=y2,
                mode='markers',
                marker=dict(size=8, color=color, opacity=0.6),
                text=momentum.index,
                hovertemplate=f'<b>%{{text}}</b><br>{period}M<br>R²: %{{x:.3f}}<br>Ann. Slope: %{{y:.2%}}<extra></extra>',
                name=f'{period}개월'
            )
        )

        # 추세선
        valid_mask2 = ~(x2.isna() | y2.isna())
        if valid_mask2.sum() > 1:
            x2_valid = x2[valid_mask2]
            y2_valid = y2[valid_mask2]
            z2 = np.polyfit(x2_valid, y2_valid, 1)
            p2 = np.poly1d(z2)
            x2_trend = np.array([x2_valid.min(), x2_valid.max()])
            y2_trend = p2(x2_trend)

            fig2.add_trace(
                go.Scatter(
                    x=x2_trend,
                    y=y2_trend,
                    mode='lines',
                    line=dict(color=color, dash='dot', width=2),
                    name=f'{period}M 추세선',
                    showlegend=False,
                    hoverinfo='skip'
                )
            )

    # 구분선
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")

    # 축 범위
    x2_all = pd.concat([momentum[f'RS{p}'] for p in periods])
    y2_all = pd.concat([momentum[f'AS{p}'] for p in periods])
    x2_range = [x2_all.min() * 0.9, x2_all.max() * 1.1]
    y2_range = [y2_all.min() * 1.1 if y2_all.min() < 0 else y2_all.min() * 0.9,
                y2_all.max() * 1.1 if y2_all.max() > 0 else y2_all.max() * 0.9]

    fig2.update_layout(
        title="Regression Momentum",
        xaxis_title="R-squared",
        yaxis_title="Annualized Slope",
        xaxis_range=x2_range,
        yaxis_range=y2_range,
        height=height,
        showlegend=True,
        hovermode='closest'
    )

    # ========== 템플릿을 사용하여 HTML 생성 ==========
    html_path = f'{dashboard_dir}/momentum.html'
    render_dashboard_html(
        title="KRX300 Momentum Analysis",
        figures=[fig1, fig2],
        chart_ids=['chart1', 'chart2'],
        output_path=html_path
    )
    print(f"  ✓ {html_path}")


def create_performance_dashboard(performance):
    """Performance 대시보드 생성 (독립된 플롯 2개를 하나의 HTML에)"""
    print("\n[2/4] Performance Dashboard 생성 중...")

    periods = settings.visualization.scatter_plot.periods
    colors = settings.visualization.scatter_plot.colors
    height = settings.dashboard.height
    dashboard_dir = settings.output.dashboard_dir
    Path(dashboard_dir).mkdir(parents=True, exist_ok=True)

    # ========== Chart 1: Sharpe Ratio (AR vs SD) ==========
    fig1 = go.Figure()
    x1_all_data = []
    y1_all_data = []

    for period, color in zip(periods, colors):
        x1 = performance[f'SD{period}']
        y1 = performance[f'AR{period}']
        x1_all_data.append(x1)
        y1_all_data.append(y1)

        # 데이터 포인트
        fig1.add_trace(
            go.Scatter(
                x=x1,
                y=y1,
                mode='markers',
                marker=dict(size=8, color=color, opacity=0.6),
                text=performance.index,
                hovertemplate=f'<b>%{{text}}</b><br>{period}M<br>SD: %{{x:.2%}}<br>AR: %{{y:.2%}}<extra></extra>',
                name=f'{period}개월'
            )
        )

        # 추세선
        valid_mask = ~(x1.isna() | y1.isna())
        if valid_mask.sum() > 1:
            x1_valid = x1[valid_mask]
            y1_valid = y1[valid_mask]
            z1 = np.polyfit(x1_valid, y1_valid, 1)
            p1 = np.poly1d(z1)
            x1_trend = np.array([x1_valid.min(), x1_valid.max()])
            y1_trend = p1(x1_trend)

            fig1.add_trace(
                go.Scatter(
                    x=x1_trend,
                    y=y1_trend,
                    mode='lines',
                    line=dict(color=color, dash='dot', width=2),
                    name=f'{period}M 추세선',
                    showlegend=False,
                    hoverinfo='skip'
                )
            )

    # 축 범위
    x1_all = pd.concat(x1_all_data)
    y1_all = pd.concat(y1_all_data)
    x1_range = [x1_all.min() * 0.9, x1_all.max() * 1.1]
    y1_range = [y1_all.min() * 1.1 if y1_all.min() < 0 else y1_all.min() * 0.9,
                y1_all.max() * 1.1 if y1_all.max() > 0 else y1_all.max() * 0.9]

    fig1.update_layout(
        title="Sharpe Ratio",
        xaxis_title="Standard Deviation",
        yaxis_title="Annualized Return",
        xaxis_range=x1_range,
        yaxis_range=y1_range,
        height=height,
        showlegend=True,
        hovermode='closest'
    )

    # ========== Chart 2: Sortino Ratio (AR vs DD) ==========
    fig2 = go.Figure()
    x2_all_data = []
    y2_all_data = []

    for period, color in zip(periods, colors):
        x2 = performance[f'DD{period}']
        y2 = performance[f'AR{period}']
        x2_all_data.append(x2)
        y2_all_data.append(y2)

        # 데이터 포인트
        fig2.add_trace(
            go.Scatter(
                x=x2,
                y=y2,
                mode='markers',
                marker=dict(size=8, color=color, opacity=0.6),
                text=performance.index,
                hovertemplate=f'<b>%{{text}}</b><br>{period}M<br>DD: %{{x:.2%}}<br>AR: %{{y:.2%}}<extra></extra>',
                name=f'{period}개월'
            )
        )

        # 추세선
        valid_mask2 = ~(x2.isna() | y2.isna())
        if valid_mask2.sum() > 1:
            x2_valid = x2[valid_mask2]
            y2_valid = y2[valid_mask2]
            z2 = np.polyfit(x2_valid, y2_valid, 1)
            p2 = np.poly1d(z2)
            x2_trend = np.array([x2_valid.min(), x2_valid.max()])
            y2_trend = p2(x2_trend)

            fig2.add_trace(
                go.Scatter(
                    x=x2_trend,
                    y=y2_trend,
                    mode='lines',
                    line=dict(color=color, dash='dot', width=2),
                    name=f'{period}M 추세선',
                    showlegend=False,
                    hoverinfo='skip'
                )
            )

    # 축 범위
    x2_all = pd.concat(x2_all_data)
    y2_all = pd.concat(y2_all_data)
    x2_range = [x2_all.min() * 0.9, x2_all.max() * 1.1]
    y2_range = [y2_all.min() * 1.1 if y2_all.min() < 0 else y2_all.min() * 0.9,
                y2_all.max() * 1.1 if y2_all.max() > 0 else y2_all.max() * 0.9]

    fig2.update_layout(
        title="Sortino Ratio",
        xaxis_title="Downside Deviation",
        yaxis_title="Annualized Return",
        xaxis_range=x2_range,
        yaxis_range=y2_range,
        height=height,
        showlegend=True,
        hovermode='closest'
    )

    # ========== 템플릿을 사용하여 HTML 생성 ==========
    html_path = f'{dashboard_dir}/performance.html'
    render_dashboard_html(
        title="KRX300 Performance Analysis",
        figures=[fig1, fig2],
        chart_ids=['chart1', 'chart2'],
        output_path=html_path
    )
    print(f"  ✓ {html_path}")


def create_correlation_network(correlation):
    """Correlation Network (VOSviewer JSON 형식)"""
    print("\n[3/4] Correlation Network 생성 중 (VOSviewer JSON)...")

    # 설정 로드
    threshold = settings.visualization.correlation_network.threshold
    layout_seed = settings.visualization.correlation_network.layout_seed
    layout_k = settings.visualization.correlation_network.layout_k

    # marginal_mean 제거하고 실제 종목들만
    corr_matrix = correlation.drop('mean', axis=0).drop('mean', axis=1)
    tickers = corr_matrix.index.tolist()

    # NetworkX 그래프 생성
    G = nx.Graph()

    for i, ticker1 in enumerate(tickers):
        G.add_node(ticker1)
        for j, ticker2 in enumerate(tickers):
            if i < j:  # 중복 방지
                corr_value = corr_matrix.loc[ticker1, ticker2]
                if corr_value > threshold:
                    G.add_edge(ticker1, ticker2, weight=corr_value)

    print(f"  노드: {G.number_of_nodes()}, 엣지: {G.number_of_edges()}")

    # 고립 노드 확인
    isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
    if isolated_nodes:
        print(f"  경고: {len(isolated_nodes)}개 고립 노드 발견 (threshold={threshold})")

    # Community detection
    communities = nx.community.greedy_modularity_communities(G)
    node_to_cluster = {}
    for cluster_id, community in enumerate(communities, 1):
        for node in community:
            node_to_cluster[node] = cluster_id

    print(f"  클러스터: {len(communities)}개")

    # Spring layout으로 좌표 계산
    pos = nx.spring_layout(G, seed=layout_seed, k=layout_k)

    # VOSviewer JSON 생성
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
    marginal_means = correlation.loc[tickers, 'mean']
    for ticker in tickers:
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
    for edge in G.edges(data=True):
        vos_data["network"]["links"].append({
            "source_id": edge[0],
            "target_id": edge[1],
            "strength": float(edge[2]['weight'])
        })

    # 클러스터 추가
    for cluster_id in range(1, len(communities) + 1):
        vos_data["network"]["clusters"].append({
            "cluster": cluster_id,
            "label": f"Cluster {cluster_id}"
        })

    # JSON 저장
    dashboard_dir = settings.output.dashboard_dir
    json_path = f'{dashboard_dir}/correlation_network.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(vos_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {json_path}")

    # VOSviewer 안내 HTML 생성 (템플릿 사용)
    html_path = f'{dashboard_dir}/correlation_network.html'
    render_data = {
        'title': 'KRX300 Correlation Network - VOSviewer',
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'n_clusters': len(communities),
        'threshold': threshold,
        'n_isolated': len(isolated_nodes)
    }
    render_html_from_template('correlation_network.html', render_data, html_path)
    print(f"  ✓ {html_path}")


def create_correlation_cluster(correlation):
    """Correlation Cluster Dendrogram with cluster assignments"""
    print("\n[4/4] Correlation Cluster Dendrogram 생성 중...")

    from scipy.cluster.hierarchy import fcluster

    width = settings.dashboard.width
    height_per_item = settings.visualization.dendrogram.height_per_item
    cluster_method = settings.visualization.dendrogram.method
    n_clusters = settings.visualization.dendrogram.n_cluster

    # marginal_mean 제거
    corr_matrix = correlation.drop('mean', axis=0).drop('mean', axis=1)
    tickers = corr_matrix.index.tolist()

    # 상관계수를 거리로 변환 (1 - correlation)
    distance_matrix = 1 - corr_matrix

    # 거리 행렬을 condensed form으로 변환
    condensed_dist = squareform(distance_matrix, checks=False)

    # Hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method=cluster_method)

    # Cluster assignments (n_clusters개의 클러스터로 분할)
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    # 클러스터별 ticker 추출
    clusters = {}
    for i in range(1, n_clusters + 1):
        cluster_tickers = [tickers[j] for j in range(len(tickers)) if cluster_labels[j] == i]
        if cluster_tickers:  # 빈 클러스터 제외
            clusters[f"Cluster_{i}"] = cluster_tickers

    print(f"  클러스터 수: {len(clusters)}개")
    print(f"  평균 클러스터 크기: {len(tickers) / len(clusters):.1f}개")

    # Plotly dendrogram
    fig = ff.create_dendrogram(
        distance_matrix.values,
        orientation='left',
        labels=tickers,
        linkagefun=lambda x: linkage(x, method=cluster_method)
    )

    height = max(800, len(corr_matrix) * height_per_item)

    fig.update_layout(
        title="KRX300 Hierarchical Clustering (Correlation-based)",
        width=width,
        height=height,
        xaxis_title="Distance (1 - Correlation)",
        yaxis_title="Stocks",
        hovermode='closest'
    )

    # 클러스터 정보를 JSON으로 저장
    dashboard_dir = settings.output.dashboard_dir

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

    # 템플릿을 사용하여 HTML 생성 (클러스터 정보 포함)
    html_path = f'{dashboard_dir}/correlation_cluster.html'
    render_html_from_template(
        'correlation_cluster.html',
        {
            'title': 'KRX300 Correlation Cluster',
            'dendrogram_html': fig.to_html(full_html=False, include_plotlyjs=False, div_id='dendrogram', config={'responsive': True}),
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
    create_momentum_dashboard(momentum)
    create_performance_dashboard(performance)
    create_correlation_network(correlation)
    create_correlation_cluster(correlation)

    dashboard_dir = settings.output.dashboard_dir

    print("\n" + "=" * 70)
    print("STEP 4 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
