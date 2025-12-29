"""설정 파일 로더 (python-box 기반)"""
from pathlib import Path
from box import Box
import yaml


# ============================================================
# Settings Loader
# ============================================================

_settings_instance = None


def _load_yaml_config():
    """YAML 파일에서 설정 로드"""
    config_path = Path(__file__).parent.parent / "settings.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def get_settings():
    """설정 인스턴스 가져오기 (싱글톤)

    Returns:
    --------
    Box
        설정 객체 (dot notation으로 접근 가능)

    Examples:
    ---------
    >>> settings = get_settings()
    >>> settings.data.n_universe
    300
    >>> settings.visualization.periods
    [12, 36, 60]
    """
    global _settings_instance
    if _settings_instance is None:
        yaml_config = _load_yaml_config()
        _settings_instance = Box(yaml_config, frozen_box=False)
    return _settings_instance


def get_config(key_path, default=None):
    """기존 코드와의 하위 호환성을 위한 함수

    Parameters:
    -----------
    key_path : str
        점(.)으로 구분된 설정 경로 (예: 'data.n_universe')
    default : any, optional
        키가 없을 때 반환할 기본값

    Returns:
    --------
    any
        설정 값 또는 기본값

    Examples:
    ---------
    >>> get_config('data.n_universe')
    300
    >>> get_config('visualization.periods')
    [12, 36, 60]
    >>> get_config('nonexistent.key', 'default')
    'default'
    """
    settings = get_settings()
    keys = key_path.split('.')
    value = settings

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def reload_settings():
    """설정 다시 로드 (테스트용)"""
    global _settings_instance
    _settings_instance = None
    return get_settings()


# ============================================================
# Convenience Aliases
# ============================================================

# 편의를 위한 별칭
settings = get_settings()
