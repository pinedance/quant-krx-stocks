"""데이터 입출력"""
from jinja2 import Environment, FileSystemLoader
import requests
import pandas as pd
from datetime import datetime
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

def get_local_list(path):
    # TODO: read local file and return data as dataframe
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
    file_loader = FileSystemLoader( base_dir )
    env = Environment(loader=file_loader)
    template = env.get_template(filename)
    return template
