"""
STEP 1: KRX300 종목 리스트 생성
- KRX API에서 KRX300 종목 리스트 가져오기
- HTML, TSV, JSON 형식으로 출력
"""

from core.io import get_list, export_dataframe_to_formats
from core.config import settings

def main():
    print("=" * 60)
    print("STEP 1: 종목 리스트 생성")
    print("=" * 60)

    # KRX300 종목 리스트 가져오기
    market = settings.data.market

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

    list_dir = settings.output.list_dir
    export_dataframe_to_formats(
        df,
        f'{list_dir}/{market}_list',
        f'{market.upper()} 종목 리스트',
        include_index=False
    )

    print("\n" + "=" * 60)
    print("STEP 1 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
