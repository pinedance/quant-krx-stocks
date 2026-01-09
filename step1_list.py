"""
STEP 1: KRX 종목 리스트 생성
- KRX API에서 KRX 종목 리스트 가져오기
- HTML, TSV, JSON 형식으로 출력
"""

from core.fetcher import get_list
from core.file import export_dataframe_to_formats, export_dataframe_to_datatable
from core.config import settings
from core.utils import print_step_header, print_completion

def main():
    print_step_header(1, "종목 리스트 생성")

    # KRX 종목 리스트 가져오기
    market = settings.stocks.list.market

    print("\n종목 리스트를 가져오는 중...")
    df = get_list(market)

    if df.empty:
        print("종목 리스트를 가져오는데 실패했습니다.")
        return

    print(f"총 {len(df)}개 종목을 가져왔습니다.")
    print("\n첫 5개 종목:")
    print(df.head())

    # 3가지 형식으로 저장
    print("\n출력 파일 생성 중...")

    list_dir = settings.output.list_dir.path
    paths = export_dataframe_to_formats(
        df,
        f'{list_dir}/{market}',
        f'{market.upper()} 종목 리스트',
        include_index=False
    )
    for path in paths.values():
        print(f"  ✓ {path}")

    # DataTables 인터랙티브 버전 추가
    print("\n인터랙티브 테이블 생성 (DataTables)...")
    datatable_path = export_dataframe_to_datatable(
        df.set_index('Code'),  # Code를 index로 설정
        f'{list_dir}/{market}',
        f'{market.upper()} 종목 리스트 - Interactive Table'
    )
    print(f"  ✓ {datatable_path}")

    print_completion(1)

if __name__ == "__main__":
    main()
