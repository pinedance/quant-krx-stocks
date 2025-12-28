"""
STEP 2: KRX300 데이터 생성 및 저장
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from core.io import get_list, get_price
from core.utils import date_before
from core.finance import annualize_rt, stdev, dsdev, get_corrMatrix
from core.models import LM
from core.config import get_config
from jinja2 import Environment, FileSystemLoader

# 템플릿 파일이 있는 디렉토리 설정
file_loader = FileSystemLoader('templates')
env = Environment(loader=file_loader)
template = env.get_template('dataframe.html')

def export_dataframe_to_formats(df, base_path, name):
    """DataFrame을 HTML, TSV, JSON 형식으로 저장"""

    # 출력 디렉토리 확인
    Path(base_path).parent.mkdir(parents=True, exist_ok=True)

    # 설정값 가져오기
    float_precision = get_config('output.html.float_precision', 6)
    large_precision = get_config('output.html.large_number_precision', 2)
    large_threshold = get_config('output.html.large_number_threshold', 1000)

    # HTML
    render_data = {
        "title": name,
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "n_row": df.shape[0],
        "n_col": df.shape[1]
    }
    html_content = template.render(render_data)
    html_path = f"{base_path}.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"  ✓ {html_path}")

    # TSV
    tsv_path = f"{base_path}.tsv"
    df.to_csv(tsv_path, sep='\t', encoding='utf-8')
    print(f"  ✓ {tsv_path}")

    # JSON
    json_path = f"{base_path}.json"
    data = {
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'shape': list(df.shape),
        'columns': df.columns.tolist(),
        'index': df.index.tolist() if isinstance(df.index, pd.DatetimeIndex) else df.index.map(str).tolist(),
        'data': df.to_dict(orient='split')['data']
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  ✓ {json_path}")


def main():
    print("=" * 70)
    print("STEP 2: KRX300 데이터 생성 및 저장")
    print("=" * 70)

    # 설정 로드
    start_years = get_config('data.start_years', 6)
    ticker_limit = get_config('data.ticker_limit', None)
    periods = get_config('analysis.periods', [12, 36, 60])
    corr_periods = get_config('analysis.correlation_periods', 12)
    data_dir = get_config('output.data_dir', 'output/data')

    # 1. KRX300 종목 리스트
    print("\n[1/7] KRX300 종목 리스트 가져오기...")
    tickers = get_list('KRX-top300')
    print(f"      총 {len(tickers)}개 종목")

    # 종목 수 제한
    if ticker_limit:
        tickers = tickers[:ticker_limit]
        print(f"      제한: {len(tickers)}개 종목만 사용")

    # 2. 가격 데이터 다운로드
    print(f"\n[2/7] 가격 데이터 다운로드 ({start_years}년치)...")
    start_date = date_before(years=start_years)
    print(f"      시작일: {start_date}")

    tickers_lst = tickers['Code'].to_list()
    closeD = get_price(tickers_lst, start_date=start_date)
    print(f"      다운로드 완료 | 종목(열): {len(closeD.shape[1])}, 날짜(행): {len(closeD.shape[0])}")

    # 3. Daily/Monthly Price DataFrame
    print("\n[3/7] Monthly Price DataFrame 생성...")
    closeM = closeD.resample('ME').last()  # 'M' → 'ME' (Month End)
    # print(closeM.shape)
    closeM.dropna(axis=1, how='any', inplace=True)
    # print(closeM.shape)
    # closeM_log = np.log(closeM.replace(0, np.nan))  # 0 방지
    closeM_log = np.log(closeM)

    print(f"      Daily:  {closeD.shape}")
    print(f"      Monthly: {closeM.shape}")

    # 4. Momentum 계산
    print("\n[4/7] Momentum 지표 계산...")
    mmtM = pd.DataFrame(index=closeM.columns)

    for i in range(1, 13):
        returns = closeM.pct_change(periods=i).iloc[-1]
        mmtM[f'{i}MR'] = returns

    mmtM['13612MR'] = (mmtM['1MR'] + mmtM['3MR'] + mmtM['6MR'] + mmtM['12MR']) / 4

    # Linear Regression (설정에서 가져온 periods 사용)
    for period in periods:
        LR = LM().fit(closeM_log, period)
        mmtM[f'AS{period}'] = ( np.exp(LR.slope * 12) - 1 )  # Monthly
        mmtM[f'RS{period}'] = LR.score

    print(f"      완료: {mmtM.shape}")

    # 5. Performance 계산
    print("\n[5/7] Performance 지표 계산...")
    pfmM = pd.DataFrame(index=closeM.columns)

    for period in periods:
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

    # 6. Correlation Matrix
    print(f"\n[6/7] Correlation Matrix 계산 (최근 {corr_periods}개월)...")
    corM = get_corrMatrix(closeM, corr_periods)
    print(f"      완료: {corM.shape}")

    # 7. 저장
    print("\n[7/7] 파일 저장 (HTML, TSV, JSON)...")

    print("  priceD:")
    export_dataframe_to_formats(closeD, f'{data_dir}/priceD', 'Daily Price (전체)')

    print("  priceM:")
    export_dataframe_to_formats(closeM, f'{data_dir}/priceM', 'Monthly Price')

    print("  momentum:")
    export_dataframe_to_formats(mmtM, f'{data_dir}/momentum', 'Momentum Indicators')

    print("  performance:")
    export_dataframe_to_formats(pfmM, f'{data_dir}/performance', 'Performance Indicators')

    print("  correlation:")
    export_dataframe_to_formats(corM, f'{data_dir}/correlation', 'Correlation Matrix')

    print("\n" + "=" * 70)
    print("STEP 2 완료!")
    print("=" * 70)
    print(f"\n생성된 파일: {data_dir}/")
    print("- priceD.{html,tsv,json}")
    print("- priceM.{html,tsv,json}")
    print("- momentum.{html,tsv,json}")
    print("- performance.{html,tsv,json}")
    print("- correlation.{html,tsv,json}")


if __name__ == "__main__":
    main()
