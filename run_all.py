"""
KRX300 프로젝트 전체 실행 스크립트
STEP 1 → STEP 2 → STEP 3 → STEP 4 → STEP 5를 순차적으로 실행합니다.
"""

import subprocess
import sys
import time


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

    print("\n생성된 파일:")
    print("  [STEP 1] output/list/")
    print("    - KRX_list.{html,tsv,json}")
    print("\n  [STEP 2] output/price/")
    print("    - priceD.{html,tsv,json}")
    print("    - priceM.{html,tsv,json}")
    print("\n  [STEP 3] output/signal/")
    print("    - momentum.{html,tsv,json}")
    print("    - performance.{html,tsv,json}")
    print("    - correlation.{html,tsv,json}")
    print("\n  [STEP 4] output/dashboard/")
    print("    - momentum.html")
    print("    - performance.html")
    print("    - correlation_network.html + .json (VOSviewer용)")
    print("    - correlation_cluster.html")
    print("\n  [STEP 5] output/")
    print("    - index.html (모든 결과물 링크)")

    print("\n다음 단계:")
    print("  1. output/index.html을 브라우저로 열어서 모든 결과물 확인")
    print("  2. Google Sheets에서 HTML table을 IMPORTHTML로 가져오기")
    print("  3. VOSviewer로 correlation_network.json 열기")
    print("  4. GitHub Pages나 웹 서버에 배포")


if __name__ == "__main__":
    main()
