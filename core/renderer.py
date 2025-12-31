"""HTML 템플릿 렌더링 모듈"""
from jinja2 import Environment, FileSystemLoader
from core.config import get_config
from core.utils import ensure_directory


def get_template(base_dir, filename):
    """
    Jinja2 템플릿을 로드합니다.

    Parameters:
    -----------
    base_dir : str
        템플릿 디렉토리 경로
    filename : str
        템플릿 파일명

    Returns:
    --------
    jinja2.Template
        로드된 템플릿 객체
    """
    file_loader = FileSystemLoader(base_dir)
    env = Environment(loader=file_loader)
    template = env.get_template(filename)
    return template


def render_html_from_template(template_name, render_data, output_path):
    """
    템플릿을 사용하여 HTML 파일을 생성합니다 (범용).

    Parameters:
    -----------
    template_name : str
        템플릿 파일 이름 (예: 'correlation_network.html')
    render_data : dict
        템플릿에 전달할 데이터
    output_path : str
        출력 파일 경로
    """
    # 출력 디렉토리 확인
    from pathlib import Path
    ensure_directory(Path(output_path).parent)

    # 템플릿 로드
    template_dir = get_config("template.base_dir")
    template = get_template(template_dir, template_name)

    # 템플릿 렌더링
    html_content = template.render(render_data)

    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def render_dashboard_html(title, figures, chart_ids, output_path):
    """
    여러 Plotly figure를 하나의 대시보드 HTML로 생성합니다.

    Parameters:
    -----------
    title : str
        대시보드 제목
    figures : list
        Plotly Figure 객체 리스트
    chart_ids : list
        각 차트의 HTML div ID 리스트
    output_path : str
        출력 파일 경로
    """
    # 출력 디렉토리 확인
    from pathlib import Path
    ensure_directory(Path(output_path).parent)

    # 템플릿 로드
    template_dir = get_config("template.base_dir")
    template = get_template(template_dir, 'dashboard.html')

    # Plotly figures를 HTML로 변환
    figures_html = [
        fig.to_html(full_html=False, include_plotlyjs=False, div_id=chart_ids[i], config={'responsive': True})
        for i, fig in enumerate(figures)
    ]

    render_data = {
        'title': title,
        'figures': figures_html
    }

    # 템플릿 렌더링
    html_content = template.render(render_data)

    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
