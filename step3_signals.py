"""
STEP 3: KRX Signals 생성 및 저장
- Momentum, Performance, Correlation 지표 계산
"""

import pandas as pd
import numpy as np
from core.file import import_dataframe_from_json, export_with_message, export_dataframe_to_datatable
from core.finance import annualize_rt, stdev, dsdev, calculate_corr_matrix
from core.models import LM
from core.signals import calculate_signals_at_date
from core.config import settings
from core.console import print_step_header, print_progress, print_completion


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

    # calculate_signals_at_date 재사용 (최신 시점 계산)
    from core.signals import calculate_signals_at_date
    mmtM, _ = calculate_signals_at_date(closeM_log, closeM, -1, include_macd=True)

    # 추가 분석 기간 지표 계산 (settings.yaml의 periods 사용)
    all_periods = sorted(set(list(range(1, 13)) + list(mnt_periods)))
    for period in all_periods:
        if len(closeM_log) >= period:
            LR = LM().fit(closeM_log, period)
            mmtM[f'AS{period}'] = (np.exp(LR.slope * 12) - 1)
            # RS3, RS6, RS12는 이미 계산되어 있음 (중복 방지)
            if period not in [3, 6, 12]:
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
    corrM = calculate_corr_matrix(closeM, corr_periods)
    print(f"      완료: {corrM.shape}")

    # 5. MACD는 이미 mmtM에 포함되어 있음 (calculate_signals_at_date에서 계산)
    print_progress(5, 5, "최종 지표 정리...")
    print(f"      완료: Momentum 지표 {mmtM.shape}, Performance 지표 {pfmM.shape}, Correlation {corrM.shape}")

    # 6. 저장
    print("\n파일 저장 (HTML, TSV, JSON)...")

    export_with_message(mmtM, f'{signal_dir}/momentum', 'Momentum Indicators')
    export_with_message(pfmM, f'{signal_dir}/performance', 'Performance Indicators')
    export_with_message(corrM, f'{signal_dir}/correlation', 'Correlation Matrix')

    # DataTables 인터랙티브 버전 추가
    print("\n인터랙티브 테이블 생성 (DataTables)...")
    mmt_dt_path = export_dataframe_to_datatable(mmtM, f'{signal_dir}/momentum', 'Momentum Indicators - Interactive Table')
    pfm_dt_path = export_dataframe_to_datatable(pfmM, f'{signal_dir}/performance', 'Performance Indicators - Interactive Table')
    print(f"  ✓ {mmt_dt_path}")
    print(f"  ✓ {pfm_dt_path}")

    print_completion(3)

if __name__ == "__main__":
    main()
