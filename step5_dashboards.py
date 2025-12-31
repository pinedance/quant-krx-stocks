"""
STEP 5: KRX300 대시보드 생성
- Momentum, Performance, Correlation 대시보드 생성
"""

import json
from core.config import settings
from core.file import import_dataframe_from_json
from core.renderer import render_dashboard_html, render_html_from_template
from core.utils import ensure_directory
from core.visualization import (
    create_monthly_momentum_chart,
    create_regression_momentum_chart,
    create_momentum_quality_vs_return_chart,
    create_momentum_strength_vs_return_chart,
    create_momentum_reliability_vs_return_chart,
    create_sharpe_ratio_chart,
    create_sortino_ratio_chart,
    create_momentum_quality_vs_sharpe_chart,
    create_momentum_quality_vs_sortino_chart,
    build_correlation_graph,
    detect_communities,
    create_vosviewer_json,
    perform_hierarchical_clustering,
    create_dendrogram_figure
)


# ============================================================
# Constants
# ============================================================

MARGINAL_MEAN_COLUMN = 'mean'


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
    dashboard_dir = settings.output.dashboard_dir.path
    ensure_directory(dashboard_dir)

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
    dashboard_dir = settings.output.dashboard_dir.path
    ensure_directory(dashboard_dir)

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
    dashboard_dir = settings.output.dashboard_dir.path
    ensure_directory(dashboard_dir)

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
    dashboard_dir = settings.output.dashboard_dir.path
    ensure_directory(dashboard_dir)

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
    print("STEP 5: KRX300 대시보드 생성")
    print("=" * 70)

    # 데이터 로드
    print("\n데이터 로딩 중...")
    signal_dir = settings.output.signal_dir.path

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
    print("STEP 5 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
