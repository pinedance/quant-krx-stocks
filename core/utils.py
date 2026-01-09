"""유틸리티 함수 모듈"""
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
import sys


# ============================================================
# 날짜 관련
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


# ============================================================
# 파일시스템 관련
# ============================================================

def ensure_directory(path):
    """
    디렉토리가 없으면 생성합니다.

    Parameters:
    -----------
    path : str or Path
        생성할 디렉토리 경로
    """
    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
# 출력 관련
# ============================================================

SEPARATOR_WIDTH = 70


def print_step_header(step_num, title, width=SEPARATOR_WIDTH):
    """
    Step 헤더를 출력합니다.

    Parameters:
    -----------
    step_num : int or str
        Step 번호
    title : str
        Step 제목
    width : int
        구분선 너비
    """
    print("=" * width)
    print(f"STEP {step_num}: {title}")
    print("=" * width)


def print_progress(current, total, message):
    """
    진행 상황을 출력합니다.

    Parameters:
    -----------
    current : int
        현재 진행 단계
    total : int
        전체 단계 수
    message : str
        진행 메시지
    """
    print(f"\n[{current}/{total}] {message}")


def print_completion(step_num, width=SEPARATOR_WIDTH):
    """
    Step 완료 메시지를 출력합니다.

    Parameters:
    -----------
    step_num : int or str
        Step 번호
    width : int
        구분선 너비
    """
    print("\n" + "=" * width)
    print(f"STEP {step_num} 완료!")
    print("=" * width)


def print_directory_tree(directory, prefix="", is_last=True):
    """
    디렉토리 트리 구조를 출력합니다.

    Parameters:
    -----------
    directory : str or Path
        출력할 디렉토리 경로
    prefix : str
        트리 구조 표현을 위한 prefix
    is_last : bool
        마지막 항목 여부
    """
    path = Path(directory)

    if not path.exists():
        return

    # 현재 디렉토리/파일 출력
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}{path.name}/")

    if path.is_dir():
        # 하위 항목 가져오기 (디렉토리 먼저, 파일 나중)
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))

        for i, item in enumerate(items):
            is_last_item = (i == len(items) - 1)
            extension = "    " if is_last else "│   "

            if item.is_dir():
                # 하위 디렉토리 재귀 호출
                print_directory_tree(item, prefix + extension, is_last_item)
            else:
                # 파일 출력
                file_connector = "└── " if is_last_item else "├── "
                print(f"{prefix}{extension}{file_connector}{item.name}")


# ============================================================
# 프로그레스 바 관련
# ============================================================

class SimpleProgressIterator:
    """로그 친화적인 간단한 프로그레스 반복자"""

    def __init__(self, iterable, desc="Progress", total=None):
        self.iterable = iterable
        self.desc = desc
        self.total = total if total is not None else len(iterable)
        self.count = 0
        self.last_percent = -1

    def __iter__(self):
        print(f"{self.desc}: 0/{self.total} (0%)")
        for item in self.iterable:
            self.count += 1
            percent = (self.count * 100) // self.total

            # 10% 단위로 출력 또는 마지막 항목
            if percent >= self.last_percent + 10 or self.count == self.total:
                print(f"{self.desc}: {self.count}/{self.total} ({percent}%)")
                self.last_percent = percent

            yield item


def smart_progress(iterable, desc="Progress", total=None):
    """
    환경에 맞는 프로그레스 표시기를 반환합니다.

    터미널(TTY)에서는 tqdm을 사용하고,
    로그/백그라운드에서는 간단한 진행률 출력을 사용합니다.

    Parameters:
    -----------
    iterable : iterable
        반복 가능한 객체
    desc : str
        진행 상황 설명
    total : int, optional
        전체 항목 수 (iterable에서 자동 추출 가능하면 생략 가능)

    Returns:
    --------
    iterator
        tqdm 또는 SimpleProgressIterator
    """
    if sys.stdout.isatty():
        # 터미널 환경: tqdm 사용
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total)
    else:
        # 로그/백그라운드 환경: 간단한 출력
        return SimpleProgressIterator(iterable, desc=desc, total=total)
