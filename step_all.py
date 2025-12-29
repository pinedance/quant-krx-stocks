"""
KRX300 프로젝트 전체 실행 스크립트
STEP 1 → STEP 2 → STEP 3 → STEP 4 → STEP 5를 순차적으로 실행합니다.
"""

import subprocess
import sys
import time
import shutil
from pathlib import Path
from core.config import settings
from core.io import print_directory_tree


def clean_output_directory():
    """output 디렉토리 삭제 및 초기화"""
    base_dir = Path(settings.output.base_dir)

    if base_dir.exists():
        print(f"기존 {base_dir} 디렉토리를 삭제합니다...")
        shutil.rmtree(base_dir)
        print(f"✓ {base_dir} 삭제 완료")

    print(f"{base_dir} 디렉토리를 생성합니다...")
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ {base_dir} 생성 완료")


def print_output_results():
    """생성된 결과물 디렉토리 트리 출력"""
    base_dir = settings.output.base_dir

    print("\n생성된 결과물:")
    print_directory_tree(base_dir, prefix="", is_last=True)


def run_step(step_name, script_name):
    """단계별 스크립트 실행"""
    print("\n" + "=" * 70)
    print(f"  {step_name}")
    print("=" * 70 + "\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            ["uv", "run", "python", script_name],
            check=True,
            capture_output=False,
            text=True
        )

        elapsed = time.time() - start_time
        print(f"\n✓ {step_name} 완료 (소요시간: {elapsed:.1f}초)")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ {step_name} 실패")
        print(f"오류: {e}")
        return False


def main():
    print("\n" + "=" * 70)
    print("  KRX300 프로젝트 전체 실행")
    print("=" * 70)

    # output 디렉토리 초기화
    print()
    clean_output_directory()

    steps = [
        ("STEP 1: KRX300 종목 리스트 생성", "step1_list.py"),
        ("STEP 2: 가격 데이터 생성", "step2_price.py"),
        ("STEP 3: Signals 생성", "step3_signals.py"),
        ("STEP 4: 대시보드 생성", "step4_dashboards.py"),
        ("STEP 5: 인덱스 페이지 생성", "step5_index.py")
    ]

    total_start = time.time()
    success_count = 0

    for step_name, script_name in steps:
        if run_step(step_name, script_name):
            success_count += 1
        else:
            print(f"\n{step_name}에서 오류가 발생하여 중단합니다.")
            sys.exit(1)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print(f"  전체 프로세스 완료!")
    print("=" * 70)
    print(f"\n완료된 단계: {success_count}/{len(steps)}")
    print(f"총 소요 시간: {total_elapsed:.1f}초 ({total_elapsed/60:.1f}분)")

    # 생성된 결과물 디렉토리 트리 출력
    print_output_results()


if __name__ == "__main__":
    main()
