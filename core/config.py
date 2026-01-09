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
    >>> settings.stocks.list.n_universe
    300
    >>> settings.visualization.periods
    [12, 36, 60]
    """
    global _settings_instance
    if _settings_instance is None:
        yaml_config = _load_yaml_config()
        _settings_instance = Box(yaml_config, frozen_box=True)
    return _settings_instance


def get_config(key_path):
    """기존 코드와의 하위 호환성을 위한 함수 (Deprecated)

    Note: 이 함수는 deprecated입니다. settings.x.y.z 패턴을 사용하세요.

    Parameters:
    -----------
    key_path : str
        점(.)으로 구분된 설정 경로 (예: 'stocks.list.n_universe')

    Returns:
    --------
    any
        설정 값 (키가 없으면 KeyError 발생)

    Examples:
    ---------
    >>> get_config('stocks.list.n_universe')
    300
    >>> get_config('visualization.periods')
    [12, 36, 60]
    """
    settings = get_settings()
    keys = key_path.split('.')
    value = settings

    for key in keys:
        value = value[key]
    return value


def reload_settings():
    """설정 다시 로드 (테스트용)"""
    global _settings_instance
    _settings_instance = None
    return get_settings()


# ============================================================
# Module-level Settings Instance
# ============================================================

# 싱글톤 인스턴스 (필요시 get_settings() 대신 사용 가능)
# 주의: 이 인스턴스는 frozen_box=True로 불변입니다
settings = get_settings()
