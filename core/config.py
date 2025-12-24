"""설정 파일 로더"""
import yaml
from pathlib import Path


class Config:
    """설정 관리 클래스"""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._load_config()
        return cls._instance

    @classmethod
    def _load_config(cls):
        """settings.yaml 로드"""
        config_path = Path(__file__).parent.parent / "settings.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            cls._config = yaml.safe_load(f)

    @classmethod
    def get(cls, key_path, default=None):
        """중첩된 키로 설정값 가져오기

        Example:
            Config.get('data.start_years') -> 6
            Config.get('visualization.colors') -> ['#FF6B6B', ...]
        """
        if cls._config is None:
            cls._load_config()

        keys = key_path.split('.')
        value = cls._config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default

            if value is None:
                return default

        return value

    @classmethod
    def reload(cls):
        """설정 파일 다시 로드"""
        cls._load_config()


# 편의 함수들
def get_config(key_path, default=None):
    """설정값 가져오기"""
    return Config.get(key_path, default)
