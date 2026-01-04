"""
STEP 3: KRX Signals 생성 및 저장
- Momentum, Performance, Correlation 지표 계산
"""

import pandas as pd
import numpy as np
from core.file import import_dataframe_from_json, export_with_message, export_dataframe_to_datatable
from core.finance import annualize_rt, stdev, dsdev, get_corr_matrix
from core.models import LM
from core.backtest import calculate_macd
from core.config import settings
from core.utils import print_step_header, print_progress, print_completion


def main():
    print_step_header(3, "KRX Signals 생성 및 저장")

    # 설정 로드
    mnt_periods = settings.signals.momentum.periods
    corr_periods = settings.signals.correlation.periods
    # input dir
    price_dir = settings.output.price_dir.path
    # output dir
    signal_dir = settings.output.signal_dir.path

    # 1. Monthly Price DataFrame 로드
    print_progress(1, 5, "Monthly Price 데이터 로드...")
    closeM = import_dataframe_from_json(f'{price_dir}/closeM.json')
    closeM.index = pd.to_datetime(closeM.index)
    closeM_log = np.log(closeM)
    print(f"      완료: {closeM.shape}")

    # 2. Momentum 지표 계산
    print_progress(2, 5, "Momentum 지표 계산...")
    mmtM = pd.DataFrame(index=closeM.columns)

    for i in range(1, 13):
        returns = closeM.pct_change(periods=i).iloc[-1]
        mmtM[f'{i}MR'] = returns

    mmtM['13612MR'] = (mmtM['1MR'] + mmtM['3MR'] + mmtM['6MR'] + mmtM['12MR']) / 4

    # Momentum (with Linear Regression)
    # 1~12개월 + mnt_periods (중복 제거 및 정렬)
    all_periods = sorted(set(list(range(1, 13)) + mnt_periods))
    for period in all_periods:
        LR = LM().fit(closeM_log, period)
        mmtM[f'AS{period}'] = (np.exp(LR.slope * 12) - 1)  # 연율화, Monthly
        mmtM[f'RS{period}'] = LR.score

    print(f"      완료: {mmtM.shape}")

    # 3. Performance 지표 계산
    print_progress(3, 5, "Performance 지표 계산...")
    pfmM = pd.DataFrame(index=closeM.columns)

    for period in mnt_periods:
        if len(closeM) >= period:
            # Annualized Return
            returns = closeM.pct_change(periods=period).iloc[-1]
            annualized_returns = annualize_rt(returns, period, 'M')
            pfmM[f'AR{period}'] = annualized_returns
            # Standard Deviation
            pfmM[f'SD{period}'] = stdev(closeM, period)
            # Downside Deviation (개선된 계산)
            pfmM[f'DD{period}'] = dsdev(closeM, period)

    print(f"      완료: {pfmM.shape}")

    # 4. Correlation Matrix 계산
    print_progress(4, 5, f"Correlation Matrix 계산 (최근 {corr_periods}개월)...")
    corrM = get_corr_matrix(closeM, corr_periods)
    print(f"      완료: {corrM.shape}")

    # 5. MACD 계산 및 momentum에 추가
    print_progress(5, 5, "MACD 오실레이터 계산...")
    macdM = calculate_macd(closeM, fast_period=12, slow_period=26, signal_period=9)
    # 마지막 행만 추출 (최신 MACD 값) - 행: 날짜, 열: 종목
    macd_latest = macdM.iloc[-1]  # Series: index=종목, value=MACD Histogram
    # momentum DataFrame에 MACD_Histogram 컬럼 추가
    mmtM['MACD_Histogram'] = macd_latest
    print(f"      완료: MACD Histogram을 momentum에 추가 ({len(mmtM)}개 종목)")

    # 6. 저장
    print("\n파일 저장 (HTML, TSV, JSON)...")

    export_with_message(mmtM, f'{signal_dir}/momentum', 'Momentum Indicators')
    export_with_message(pfmM, f'{signal_dir}/performance', 'Performance Indicators')
    export_with_message(corrM, f'{signal_dir}/correlation', 'Correlation Matrix')

    # DataTables 인터랙티브 버전 추가
    print("\n인터랙티브 테이블 생성 (DataTables)...")
    export_dataframe_to_datatable(mmtM, f'{signal_dir}/momentum', 'Momentum Indicators - Interactive Table')
    export_dataframe_to_datatable(pfmM, f'{signal_dir}/performance', 'Performance Indicators - Interactive Table')

    print_completion(3)

if __name__ == "__main__":
    main()
