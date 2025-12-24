"""데이터 입출력"""
import requests
import pandas as pd
from datetime import datetime
import FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from core.config import get_config


def get_krx300_list(trade_date=None):
    """
    KRX300 종목 리스트를 KRX API에서 가져옵니다.
    """
    if trade_date is None:
        trade_date = datetime.now().strftime('%Y%m%d')

    url = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201010105',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'X-Requested-With': 'XMLHttpRequest'
    }

    payload = {
        'bld': 'dbms/MDC/STAT/standard/MDCSTAT00601',
        'locale': 'ko_KR',
        'tboxindIdx_finder_equidx0_1': 'KRX 300',
        'indIdx': '5',
        'indIdx2': '300',
        'codeNmindIdx_finder_equidx0_1': 'KRX 300',
        'param1indIdx_finder_equidx0_1': '',
        'trdDd': trade_date,
        'money': '3',
        'csvxls_isNo': 'false'
    }

    response = requests.post(url, headers=headers, data=payload)
    response.raise_for_status()
    data = response.json()

    if 'output' in data:
        return pd.DataFrame(data['output'])
    else:
        raise ValueError("No data found in response")


def get_list(index_name='KRX300', trade_date=None):
    """지수 구성 종목 리스트를 가져옵니다."""
    if index_name != 'KRX300':
        raise ValueError(f"Unsupported index: {index_name}")

    df = get_krx300_list(trade_date)

    if 'ISU_SRT_CD' in df.columns:
        return df['ISU_SRT_CD'].tolist()
    elif 'ISU_CD' in df.columns:
        return df['ISU_CD'].tolist()
    else:
        raise ValueError("Could not find ticker column in response")


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
    use_parallel = get_config('data.parallel_downloads', True)
    data_source = get_config('data.data_source', None)
    max_workers = os.cpu_count()

    price_data = {}

    # 병렬 다운로드
    if use_parallel and len(tickers) > 5:
        print(f"병렬 다운로드 모드 (workers={max_workers})")

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
        print("순차 다운로드 모드")

        for ticker in tqdm(tickers, desc="Downloading"):
            ticker_with_source = f'{data_source}:{ticker}' if data_source else ticker
            try:
                df = fdr.DataReader(ticker_with_source, start_date, end_date)
                if not df.empty:
                    df.columns = [col.lower() for col in df.columns]
                    price_data[ticker] = df
            except Exception:
                continue

    return price_data
