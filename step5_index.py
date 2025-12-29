"""
STEP 5: 결과물 인덱스 페이지 생성
- output 디렉토리의 모든 결과물을 정리한 index.html 생성
"""

from pathlib import Path
from datetime import datetime
from core.config import get_config
from core.io import render_html_from_template


def scan_output_directory(base_dir):
    """output 디렉토리를 스캔하여 파일 목록 생성

    Returns:
    --------
    list of dict
        섹션 정보 (title, description, items)
    """
    base_path = Path(base_dir)

    # 섹션 정의 (순서대로)
    section_configs = [
        {
            'dir': 'list',
            'title': '1. Stock List',
            'description': 'KRX 시장의 전체 종목 리스트 (시가총액 순)',
            'items': [
                {
                    'name': 'KRX_list',
                    'base': 'KRX_list',
                    'description': f'{get_config("data.market", "KRX")} 전체 종목 정보'
                }
            ]
        },
        {
            'dir': 'price',
            'title': '2. Price Data',
            'description': '종목별 가격 데이터 (일별/월별)',
            'items': [
                {
                    'name': 'Daily Price',
                    'base': 'priceD',
                    'description': '일별 종가 데이터 (전체 다운로드 종목)'
                },
                {
                    'name': 'Monthly Price',
                    'base': 'priceM',
                    'description': f'월별 종가 데이터 (상위 {get_config("data.n_universe", 300)}개 종목)'
                }
            ]
        },
        {
            'dir': 'signal',
            'title': '3. Signal Indicators',
            'description': '모멘텀, 성과, 상관관계 지표',
            'items': [
                {
                    'name': 'Momentum',
                    'base': 'momentum',
                    'description': '월별 수익률, 평균 모멘텀, 회귀 모멘텀 지표'
                },
                {
                    'name': 'Performance',
                    'base': 'performance',
                    'description': 'Sharpe Ratio, Sortino Ratio 성과 지표'
                },
                {
                    'name': 'Correlation',
                    'base': 'correlation',
                    'description': '종목 간 상관계수 행렬 및 marginal mean'
                }
            ]
        },
        {
            'dir': 'dashboard',
            'title': '4. Interactive Dashboards',
            'description': 'Plotly 인터랙티브 시각화 및 네트워크 분석',
            'items': [
                {
                    'name': 'Momentum Dashboard',
                    'files': ['momentum.html'],
                    'description': 'Monthly Momentum, Regression Momentum 산점도'
                },
                {
                    'name': 'Performance Dashboard',
                    'files': ['performance.html'],
                    'description': 'Sharpe Ratio, Sortino Ratio 산점도'
                },
                {
                    'name': 'Correlation Network',
                    'files': ['correlation_network.html', 'correlation_network.json'],
                    'description': 'VOSviewer 네트워크 분석 (JSON 파일 포함)'
                },
                {
                    'name': 'Correlation Cluster',
                    'files': ['correlation_cluster.html'],
                    'description': 'Hierarchical Clustering 덴드로그램'
                }
            ]
        }
    ]

    sections = []
    total_files = 0
    total_items = 0

    for config in section_configs:
        section_dir = base_path / config['dir']

        if not section_dir.exists():
            continue

        section = {
            'title': config['title'],
            'description': config.get('description', ''),
            'data_items': []
        }

        for item_config in config['items']:
            item = {
                'name': item_config['name'],
                'description': item_config.get('description', ''),
                'files': []
            }

            # Dashboard는 files 리스트를 직접 지정
            if 'files' in item_config:
                for filename in item_config['files']:
                    file_path = section_dir / filename
                    if file_path.exists():
                        ext = filename.split('.')[-1]
                        item['files'].append({
                            'path': f"{config['dir']}/{filename}",
                            'label': ext.upper(),
                            'type': ext
                        })
                        total_files += 1
            # 기타는 base 이름으로 html, tsv, json 찾기
            else:
                base_name = item_config['base']
                for ext in ['html', 'tsv', 'json']:
                    file_path = section_dir / f"{base_name}.{ext}"
                    if file_path.exists():
                        item['files'].append({
                            'path': f"{config['dir']}/{base_name}.{ext}",
                            'label': ext.upper(),
                            'type': ext
                        })
                        total_files += 1

            if item['files']:
                section['data_items'].append(item)
                total_items += 1

        if section['data_items']:
            sections.append(section)

    return sections, total_files, total_items


def main():
    print("=" * 70)
    print("STEP 5: 결과물 인덱스 페이지 생성")
    print("=" * 70)

    # 설정 로드
    base_dir = get_config('output.base_dir', 'output')
    project_name = get_config('project.name', 'KRX300 Quantitative Analysis')

    # 파일 스캔
    print("\n[1/2] 파일 스캔 중...")
    sections, total_files, total_items = scan_output_directory(base_dir)

    print(f"      섹션: {len(sections)}개")
    print(f"      항목: {total_items}개")
    print(f"      파일: {total_files}개")

    # 인덱스 HTML 생성
    print("\n[2/2] 인덱스 페이지 생성 중...")
    render_data = {
        'title': 'KRX300 Analysis Results',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'project_name': project_name,
        'n_sections': len(sections),
        'n_items': total_items,
        'n_files': total_files,
        'sections': sections
    }

    output_path = f'{base_dir}/index.html'
    render_html_from_template('index.html', render_data, output_path)
    print(f"  ✓ {output_path}")

    print("\n" + "=" * 70)
    print("STEP 5 완료!")
    print(f"브라우저에서 {output_path} 파일을 열어보세요.")
    print("=" * 70)


if __name__ == "__main__":
    main()
