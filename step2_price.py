"""
STEP 2: KRX300 데이터 생성 및 저장
"""

import pandas as pd
from core.io import import_dataframe_from_json, get_price, export_dataframe_to_formats
from core.utils import date_before
from core.config import settings


def main():
    print("=" * 70)
    print("STEP 2: KRX300 데이터 생성 및 저장")
    print("=" * 70)

    # 설정 로드
    n_universe = settings.data.n_universe
    n_try = n_universe + 100
    price_periods = settings.data.price.periods
    # input dir
    list_dir = settings.output.list_dir
    # output dir
    price_dir = settings.output.price_dir

    # 1. 종목 리스트
    print("\n[1/4] 종목 리스트 가져오기...")
    market = settings.data.market
    tickers = import_dataframe_from_json(f'{list_dir}/{market}_list.json')
    print(f"      총 {len(tickers)}개 종목")

    # 2. 가격 데이터 다운로드
    print(f"\n[2/4] 가격 데이터 다운로드 ({price_periods}개월치)...")
    start_date = date_before(months=price_periods)
    print(f"      시작일: {start_date}")
    print( tickers )
    tickers_lst = tickers['Code'].to_list()[:n_try]
    closeD = get_price(tickers_lst, start_date=start_date)
    print(f"      다운로드 완료 | 종목(열): {closeD.shape[1]}, 날짜(행): {closeD.shape[0]}")

    # 3. Daily/Monthly Price DataFrame
    print("\n[3/4] Monthly Price DataFrame 생성...")
    closeM = closeD.resample('ME').last()  # 'M' → 'ME' (Month End)
    closeM.dropna(axis=1, how='any', inplace=True)
    # ticker 개수 조정 
    new_columns = closeM.columns[:n_universe]
    # 마지막 행 제거 (현재 가격 => 이전 달 종가)
    closeM = closeM[new_columns].iloc[:-1]    

    print(f"      Daily:  {closeD.shape}")
    print(f"      Monthly: {closeM.shape}")

    # 4. 저장
    print("\n[4/4] 파일 저장 (HTML, TSV, JSON)...")

    print("  closeD:")
    export_dataframe_to_formats(closeD, f'{price_dir}/closeD', 'Daily Price (전체)')

    print("  closeM:")
    export_dataframe_to_formats(closeM, f'{price_dir}/closeM', 'Monthly Price')

    print("\n" + "=" * 70)
    print("STEP 2 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
