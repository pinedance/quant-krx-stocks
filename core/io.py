"""데이터 입출력"""
from jinja2 import Environment, FileSystemLoader
import requests
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from core.config import get_config


def _get_krx_list():
    cols = ["Code", "ISU_CD", "Name", "Market", "Volume", "Amount", "Marcap", "Stocks", "MarketId"]
    return fdr.StockListing("KRX")[cols].sort_values(by='Marcap', ascending=False)

def get_list(market="KRX"):
    """지수 구성 종목 리스트를 가져옵니다."""
    if market == 'KRX':
        df = _get_krx_list()
    else:
        raise ValueError(f"Unsupported market: {market}")
    return df

def _download_single_ticker(ticker, start_date, end_date, data_source=None):
    """단일 종목 다운로드 (병렬 처리용)"""
    try:
        ticker_with_source = f'{data_source}:{ticker}' if data_source else ticker
        df = fdr.DataReader(ticker_with_source, start_date, end_date)
        if not df.empty:
            df.columns = [col.lower() for col in df.columns]
            return ticker, df
        else:
            return ticker, None
    except Exception:
        return ticker, None


def get_price(tickers, start_date=None, end_date=None):
    """
    종목들의 가격 데이터를 가져옵니다 (병렬 처리).

    Parameters:
    -----------
    tickers : list
        종목 코드 리스트
    start_date : str
        시작일 (YYYY-MM-DD 형식)
    end_date : str, optional
        종료일 (YYYY-MM-DD 형식). None이면 오늘 날짜

    Returns:
    --------
    dict
        ticker를 key로 하고 DataFrame을 value로 하는 딕셔너리
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # 설정 로드
    download_method = get_config('data.download_method', "parallel")
    data_source = get_config('data.data_source', None)
    max_workers = os.cpu_count()

    if (download_method == "batch") or (download_method == "parallel" and len(tickers) <= 10):
        print("      배치 다운로드 모드")
        tickers_str = ",".join(tickers)
        df = fdr.DataReader(tickers_str, start_date, end_date)
        if not df.empty:
            df.columns = [col.lower() for col in df.columns]
        # print(df)
        closeD = df
        return closeD

    price_data = {}
    # 병렬 다운로드
    if download_method == "parallel" and len(tickers) > 10:
        print(f"      병렬 다운로드 모드 (workers={max_workers})")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_download_single_ticker, ticker, start_date, end_date, data_source): ticker
                for ticker in tickers
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
                ticker, df = future.result()
                if df is not None:
                    price_data[ticker] = df
    # 순차 다운로드
    else:
        print("      순차 다운로드 모드")
        for ticker in tqdm(tickers, desc="Downloading"):
            try:
                ticker, df = _download_single_ticker(ticker, start_date, end_date, data_source)
                if df:
                    price_data[ticker] = df
            except Exception:
                continue

    close_series = {
        ticker: df['close']
        for ticker, df in price_data.items()
        if 'close' in df.columns
    }
    closeD = pd.DataFrame(close_series)

    return closeD

def get_template(base_dir, filename):
    """Jinja2 템플릿 로드"""
    file_loader = FileSystemLoader(base_dir)
    env = Environment(loader=file_loader)
    template = env.get_template(filename)
    return template

def import_dataframe_from_json(json_path):
    """JSON 파일에서 DataFrame 가져오기"""

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(
        data['data'],
        index=data['index'],
        columns=data['columns']
    )
    return df


def export_dataframe_to_html(df, base_path, name):
    """DataFrame을 HTML로 저장

    Parameters:
    -----------
    df : pd.DataFrame
        저장할 DataFrame
    base_path : str
        파일 경로 (확장자 제외)
    name : str
        제목
    """
    # 출력 디렉토리 확인
    Path(base_path).parent.mkdir(parents=True, exist_ok=True)

    # 템플릿 로드
    template_dir = get_config("template.base_dir")
    template = get_template(template_dir, 'dataframe.html')

    # HTML 렌더링 데이터
    render_data = {
        "title": name,
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "n_row": df.shape[0],
        "n_col": df.shape[1],
        "dataframe": df.to_html(index=True, escape=False)
    }

    html_content = template.render(render_data)
    html_path = f"{base_path}.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"  ✓ {html_path}")


def export_dataframe_to_tsv(df, base_path, include_index=True):
    """DataFrame을 TSV로 저장

    Parameters:
    -----------
    df : pd.DataFrame
        저장할 DataFrame
    base_path : str
        파일 경로 (확장자 제외)
    include_index : bool
        index 포함 여부
    """
    # 출력 디렉토리 확인
    Path(base_path).parent.mkdir(parents=True, exist_ok=True)

    tsv_path = f"{base_path}.tsv"
    df.to_csv(tsv_path, sep='\t', encoding='utf-8', index=include_index)
    print(f"  ✓ {tsv_path}")


def export_dataframe_to_json(df, base_path):
    """DataFrame을 JSON으로 저장

    Parameters:
    -----------
    df : pd.DataFrame
        저장할 DataFrame
    base_path : str
        파일 경로 (확장자 제외)
    """
    # 출력 디렉토리 확인
    Path(base_path).parent.mkdir(parents=True, exist_ok=True)

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


def export_dataframe_to_formats(df, base_path, name, include_index=True):
    """DataFrame을 HTML, TSV, JSON 형식으로 저장 (wrapper 함수)

    Parameters:
    -----------
    df : pd.DataFrame
        저장할 DataFrame
    base_path : str
        파일 경로 (확장자 제외)
    name : str
        제목
    include_index : bool
        TSV 저장시 index 포함 여부
    """
    export_dataframe_to_html(df, base_path, name)
    export_dataframe_to_tsv(df, base_path, include_index)
    export_dataframe_to_json(df, base_path)


def render_html_from_template(template_name, render_data, output_path):
    """템플릿을 사용하여 HTML 생성 (범용)

    Parameters:
    -----------
    template_name : str
        템플릿 파일 이름 (예: 'correlation_network.html')
    render_data : dict
        템플릿에 전달할 데이터
    output_path : str
        출력 파일 경로
    """
    from pathlib import Path

    # 출력 디렉토리 확인
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 템플릿 로드
    template_dir = get_config("template.base_dir")
    template = get_template(template_dir, template_name)

    # 템플릿 렌더링
    html_content = template.render(render_data)

    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def render_dashboard_html(title, figures, chart_ids, output_path):
    """여러 Plotly figure를 하나의 HTML로 생성"""
    from pathlib import Path

    # 출력 디렉토리 확인
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 템플릿 로드
    template_dir = get_config("template.base_dir")
    template = get_template(template_dir, 'dashboard.html')

    # Plotly figures를 HTML로 변환
    figures_html = [
        fig.to_html(full_html=False, include_plotlyjs=False, div_id=chart_ids[i], config={'responsive': True})
        for i, fig in enumerate(figures)
    ]

    render_data = {
        'title': title,
        'figures': figures_html
    }

    # 템플릿 렌더링
    html_content = template.render(render_data)

    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def print_directory_tree(directory, prefix="", is_last=True):
    """디렉토리 트리 구조 출력

    Parameters:
    -----------
    directory : str or Path
        출력할 디렉토리 경로
    prefix : str
        트리 구조 표현을 위한 prefix
    is_last : bool
        마지막 항목 여부
    """
    path = Path(directory)

    if not path.exists():
        return

    # 현재 디렉토리/파일 출력
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}{path.name}/")

    if path.is_dir():
        # 하위 항목 가져오기 (디렉토리 먼저, 파일 나중)
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))

        for i, item in enumerate(items):
            is_last_item = (i == len(items) - 1)
            extension = "    " if is_last else "│   "

            if item.is_dir():
                # 하위 디렉토리 재귀 호출
                print_directory_tree(item, prefix + extension, is_last_item)
            else:
                # 파일 출력
                file_connector = "└── " if is_last_item else "├── "
                print(f"{prefix}{extension}{file_connector}{item.name}")
