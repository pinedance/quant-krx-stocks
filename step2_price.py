"""
STEP 2: KRX300 데이터 생성 및 저장
"""

from core.fetcher import get_price
from core.file import import_dataframe_from_json, export_with_message, export_dataframe_to_datatable
from core.utils import date_before, print_step_header, print_progress, print_completion
from core.config import settings


def main():
    print_step_header(2, "KRX300 데이터 생성 및 저장")

    # 설정 로드
    n_universe = settings.data.n_universe
    n_try = n_universe + settings.data.n_buffer
    price_periods = settings.data.price.periods
    # input dir
    list_dir = settings.output.list_dir.path
    # output dir
    price_dir = settings.output.price_dir.path

    # 1. 종목 리스트
    print_progress(1, 4, "종목 리스트 가져오기...")
    market = settings.data.market
    tickers = import_dataframe_from_json(f'{list_dir}/{market}.json')
    print(f"      총 {len(tickers)}개 종목")

    # 2. 가격 데이터 다운로드
    print_progress(2, 4, f"가격 데이터 다운로드 ({price_periods}개월치)...")
    start_date = date_before(months=price_periods)
    print(f"      시작일: {start_date}")
    tickers_lst = tickers['Code'].to_list()[:n_try]
    closeD = get_price(tickers_lst, start_date=start_date)
    print(f"      다운로드 완료 | 종목(열): {closeD.shape[1]}, 날짜(행): {closeD.shape[0]}")

    # 3. Daily/Monthly Price DataFrame
    print_progress(3, 4, "Monthly Price DataFrame 생성...")
    _closeM = closeD.resample('ME').last()  # 'M' → 'ME' (Month End)

    # min_periods: 최소 개월 수 (전후 1개월씩 추가)
    min_periods = 1 + settings.data.price.min_periods + 1

    # 처음 min_periods 개월 동안 데이터가 완전한 종목만 선택 (최근 상장 종목 제외)
    _closeM_filtered = _closeM.iloc[:min_periods].dropna(axis=1, how='any')

    # ticker 개수를 n_universe로 조정
    n_universe_updated = min(len(_closeM_filtered.columns), n_universe)
    selected_tickers = _closeM_filtered.columns[:n_universe_updated]

    # 선택된 종목의 전체 기간 데이터 (마지막 행 제거: 현재 가격 기준이 '이전 달 종가'이므로)
    closeM = _closeM[selected_tickers].iloc[:-1]

    print(f"      Daily:  {closeD.shape}")
    print(f"      Monthly: {closeM.shape}")

    # 4. 저장
    print_progress(4, 4, "파일 저장 (HTML, TSV, JSON)...")

    export_with_message(closeD, f'{price_dir}/closeD', 'Daily Price (전체)')
    export_with_message(closeM, f'{price_dir}/closeM', 'Monthly Price')

    # DataTables 인터랙티브 버전 추가
    print("\n인터랙티브 테이블 생성 (DataTables)...")
    export_dataframe_to_datatable(closeD, f'{price_dir}/closeD', 'Daily Price - Interactive Table')
    export_dataframe_to_datatable(closeM, f'{price_dir}/closeM', 'Monthly Price - Interactive Table')

    print_completion(2)


if __name__ == "__main__":
    main()
