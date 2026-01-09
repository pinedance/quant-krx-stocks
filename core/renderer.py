"""HTML 템플릿 렌더링 모듈 - 순수 렌더링만 담당"""
from jinja2 import Environment, FileSystemLoader
from core.config import get_config


# ============================================================
# Jinja2 Environment 싱글톤
# ============================================================

_jinja_env = None


def _get_jinja_env():
    """
    Jinja2 Environment를 싱글톤으로 반환합니다.

    Returns:
    --------
    jinja2.Environment
        Jinja2 환경 객체
    """
    global _jinja_env
    if _jinja_env is None:
        template_dir = get_config("template.base_dir")
        _jinja_env = Environment(loader=FileSystemLoader(template_dir))
    return _jinja_env


# ============================================================
# 템플릿 렌더링 함수
# ============================================================

def render_template(template_name, data):
    """
    템플릿을 렌더링하여 문자열로 반환합니다.

    Parameters:
    -----------
    template_name : str
        템플릿 파일 이름 (예: 'index.html', 'telegram_message.txt')
    data : dict
        템플릿에 전달할 데이터

    Returns:
    --------
    str
        렌더링된 콘텐츠
    """
    env = _get_jinja_env()
    template = env.get_template(template_name)
    return template.render(data)
