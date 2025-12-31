"""
STEP 3: KRX300 Signals 생성 및 저장
- Momentum, Performance, Correlation 지표 계산
"""

import pandas as pd
import numpy as np
from core.io import import_dataframe_from_json, export_dataframe_to_formats
from core.finance import annualize_rt, stdev, dsdev, get_corrMatrix
from core.models import LM
from core.config import settings


def main():
    print("=" * 70)
    print("STEP 3: KRX300 Signals 생성 및 저장")
    print("=" * 70)

    # 설정 로드
    mnt_periods = settings.signals.momentum.periods
    corr_periods = settings.signals.correlation.periods
    # input dir
    price_dir = settings.output.price_dir
    # output dir
    signal_dir = settings.output.signal_dir

    # 1. Monthly Price DataFrame 로드
    print("\n[1/4] Monthly Price 데이터 로드...")
    closeM = import_dataframe_from_json(f'{price_dir}/closeM.json')
    closeM.index = pd.to_datetime(closeM.index)
    closeM_log = np.log(closeM)
    print(f"      완료: {closeM.shape}")

    # 2. Momentum 지표 계산
    print("\n[2/4] Momentum 지표 계산...")
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
    print("\n[3/4] Performance 지표 계산...")
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
    print(f"\n[4/4] Correlation Matrix 계산 (최근 {corr_periods}개월)...")
    corM = get_corrMatrix(closeM, corr_periods)
    print(f"      완료: {corM.shape}")

    # 5. 저장
    print("\n파일 저장 (HTML, TSV, JSON)...")

    print("  momentum:")
    export_dataframe_to_formats(mmtM, f'{signal_dir}/momentum', 'Momentum Indicators')

    print("  performance:")
    export_dataframe_to_formats(pfmM, f'{signal_dir}/performance', 'Performance Indicators')

    print("  correlation:")
    export_dataframe_to_formats(corM, f'{signal_dir}/correlation', 'Correlation Matrix')

    print("\n" + "=" * 70)
    print("STEP 3 완료!")
    print("=" * 70)

if __name__ == "__main__":
    main()
