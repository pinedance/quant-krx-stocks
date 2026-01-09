"""
STEP 2: KRX 데이터 생성 및 저장
"""
from datetime import datetime
import pytz
from core.fetcher import get_price
from core.file import import_dataframe_from_json, export_with_message, export_dataframe_to_datatable
from core.utils import date_before
from core.console import print_step_header, print_progress, print_completion
from core.config import settings


def main():
    print_step_header(2, "KRX 데이터 생성 및 저장")

    # 설정 로드
    n_universe = settings.stocks.list.n_universe
    n_try = n_universe + settings.stocks.list.n_buffer
    price_periods = settings.stocks.price.periods
    # input dir
    list_dir = settings.output.list_dir.path
    # output dir
    price_dir = settings.output.price_dir.path

    # 1. 종목 리스트
    print_progress(1, 4, "종목 리스트 가져오기...")
    market = settings.stocks.list.market
    tickers = import_dataframe_from_json(f'{list_dir}/{market}.json')
    print(f"      총 {len(tickers)}개 종목")

    # 2. 가격 데이터 다운로드
    print_progress(2, 4, f"가격 데이터 다운로드 ({price_periods}개월치)...")
    start_date = date_before(months=price_periods)
    print(f"      시작일: {start_date}")

    # n_try만큼만 선택
    tickers_subset = tickers.head(n_try)

    # 위험 종목 필터링
    exclude_risky = settings.stocks.price.exclude_risky_stocks
    if exclude_risky:
        risky_columns = ['관리종목여부', '거래정지여부', '투자주의여부', '투자경고여부', '투자위험여부']
        # 컬럼이 존재하는지 확인
        existing_risky_columns = [col for col in risky_columns if col in tickers_subset.columns]

        if existing_risky_columns:
            total_count = len(tickers_subset)
            # 위험 종목 제외: 모든 플래그가 False인 종목만 선택
            mask = True
            for col in existing_risky_columns:
                mask = mask & (~tickers_subset[col])

            # 제외된 종목 정보
            excluded = tickers_subset[~mask]
            tickers_subset = tickers_subset[mask]

            excluded_count = len(excluded)
            remaining_count = len(tickers_subset)

            print(f"      위험 종목 필터링:")
            print(f"      - 전체: {total_count}개")
            print(f"      - 제외: {excluded_count}개")
            print(f"      - 다운로드 대상: {remaining_count}개")

            if excluded_count > 0 and excluded_count <= 20:
                # 제외된 종목이 20개 이하면 전체 출력
                excluded_list = [f"{row['Code']}({row['Name']})"
                               for _, row in excluded.iterrows()
                               if 'Code' in row and 'Name' in row]
                print(f"      - 제외된 종목: {', '.join(excluded_list)}")
            elif excluded_count > 20:
                # 20개 초과면 일부만 출력
                excluded_list = [f"{row['Code']}({row['Name']})"
                               for _, row in excluded.head(10).iterrows()
                               if 'Code' in row and 'Name' in row]
                print(f"      - 제외된 종목 (일부): {', '.join(excluded_list)} ...")
        else:
            print(f"      위험 종목 컬럼이 없어 필터링을 건너뜁니다.")

    # Code 리스트로 변환
    tickers_lst = tickers_subset['Code'].tolist()
    closeD = get_price(tickers_lst, start_date=start_date)
    print(f"      다운로드 완료 | 종목(열): {closeD.shape[1]}, 날짜(행): {closeD.shape[0]}")

    # 3. Daily/Monthly Price DataFrame
    print_progress(3, 4, "Monthly Price DataFrame 생성...")
    _closeM = closeD.resample('ME').last()  # 'M' → 'ME' (Month End)

    # min_periods: 최소 개월 수 (전후 1개월씩 추가)
    min_periods = 1 + settings.stocks.price.min_periods + 1

    # 처음 min_periods 개월 동안 데이터가 완전한 종목만 선택 (최근 상장 종목 제외)
    _closeM_filtered = _closeM.iloc[(-1*min_periods):].dropna(axis=1, how='any')

    # ticker 개수를 n_universe로 조정
    print(f"      목표 종목 개수: {n_universe}, 최소 데이터 충족 종목 개수: {len(_closeM_filtered.columns)}")
    n_universe_updated = min(len(_closeM_filtered.columns), n_universe)
    selected_tickers = _closeM_filtered.columns[:n_universe_updated]

    # 선택된 종목의 전체 기간 데이터 (현재 날짜 이전 데이터만 포함)
    kst = pytz.timezone('Asia/Seoul')   # 현재 날짜(한국 시간 기준)
    current_date = datetime.now(kst).replace(tzinfo=None)  # timezone 제거 (pandas index와 비교 위해)
    closeM = _closeM.loc[_closeM.index <= current_date, selected_tickers]

    print(f"      Daily:  {closeD.shape}")
    print(f"      Monthly: {closeM.shape}")

    # 4. 저장
    print_progress(4, 4, "파일 저장 (HTML, TSV, JSON)...")

    export_with_message(closeD, f'{price_dir}/closeD', 'Daily Price (전체)')
    export_with_message(closeM, f'{price_dir}/closeM', 'Monthly Price')

    # DataTables 인터랙티브 버전 추가
    print("\n인터랙티브 테이블 생성 (DataTables)...")
    closeD_dt_path = export_dataframe_to_datatable(closeD, f'{price_dir}/closeD', 'Daily Price - Interactive Table')
    closeM_dt_path = export_dataframe_to_datatable(closeM, f'{price_dir}/closeM', 'Monthly Price - Interactive Table')
    print(f"  ✓ {closeD_dt_path}")
    print(f"  ✓ {closeM_dt_path}")

    print_completion(2)


if __name__ == "__main__":
    main()
