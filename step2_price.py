"""
STEP 2: KRX300 데이터 생성 및 저장
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from core.io import get_local_list, get_price, get_template
from core.utils import date_before
from core.finance import annualize_rt, stdev, dsdev, get_corrMatrix
from core.models import LM
from core.config import get_config

# 템플릿 파일이 있는 디렉토리 설정
template = get_template(get_config(template.base_dir), 'dataframe.html')

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
    n_universe = get_config('data.n_universe', 300)
    n_try = n_universe + 100
    price_periods = get_config('data.price.periods', 63)
    # input dir
    list_dir = get_config('output.list_dir')
    # output dir
    price_dir = get_config('output.price_dir', 'output/price')

    # 1. 종목 리스트
    print("\n[1/7] 종목 리스트 가져오기...")
    tickers = get_local_list(list_dir)
    print(f"      총 {len(tickers)}개 종목")

    # 2. 가격 데이터 다운로드
    print(f"\n[2/7] 가격 데이터 다운로드 ({price_periods}년치)...")
    start_date = date_before(months=price_periods)
    print(f"      시작일: {start_date}")

    tickers_lst = tickers['Code'].to_list()[:n_try]
    closeD = get_price(tickers_lst, start_date=start_date)
    print(f"      다운로드 완료 | 종목(열): {len(closeD.shape[1])}, 날짜(행): {len(closeD.shape[0])}")

    # 3. Daily/Monthly Price DataFrame
    print("\n[3/7] Monthly Price DataFrame 생성...")
    closeM = closeD.resample('ME').last()  # 'M' → 'ME' (Month End)
    # print(closeM.shape)
    closeM.dropna(axis=1, how='any', inplace=True)
    new_columns = closeM.columns[:n_universe]
    closeM = closeM[new_columns]
    # print(closeM.shape)
    # closeM_log = np.log(closeM.replace(0, np.nan))  # 0 방지
    closeM_log = np.log(closeM)

    print(f"      Daily:  {closeD.shape}")
    print(f"      Monthly: {closeM.shape}")

    # 7. 저장
    print("\n[7/7] 파일 저장 (HTML, TSV, JSON)...")

    print("  priceD:")
    export_dataframe_to_formats(closeD, f'{data_dir}/priceD', 'Daily Price (전체)')

    print("  priceM:")
    export_dataframe_to_formats(closeM, f'{data_dir}/priceM', 'Monthly Price')


if __name__ == "__main__":
    main()
