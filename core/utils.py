"""유틸리티 함수 모듈 (Deprecated Wrapper)

이 모듈은 하위 호환성을 위해 유지됩니다.
새 코드에서는 아래 모듈을 직접 import하세요:
- core.console: 콘솔 출력 및 프로그레스 표시
- core.file: 파일시스템 유틸리티
"""
from datetime import datetime
from dateutil.relativedelta import relativedelta

# 하위 호환성을 위한 재export
from core.console import (
    print_step_header,
    print_progress,
    print_completion,
    print_directory_tree,
    smart_progress,
    SimpleProgressIterator
)
from core.file import ensure_directory


# ============================================================
# 날짜 관련 (이 모듈에 유지)
# ============================================================

def date_before(years=0, months=0, days=0, date_format='%Y-%m-%d'):
    """
    현재 날짜 기준으로 과거 날짜를 계산합니다.

    Parameters:
    -----------
    years : int
        년수
    months : int
        개월수
    days : int
        일수
    date_format : str
        반환할 날짜 형식

    Returns:
    --------
    str
        지정된 형식의 날짜 문자열
    """
    today = datetime.now()
    past_date = today - relativedelta(years=years, months=months, days=days)
    return past_date.strftime(date_format)
