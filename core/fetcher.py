"""주식 데이터 수집 모듈"""
from datetime import datetime
import pandas as pd
import FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from core.config import get_config


def _get_krx_list():
    """KRX 전체 종목 리스트를 가져옵니다 (내부 함수)"""
    cols = ["Code", "ISU_CD", "Name", "Market", "Volume", "Amount", "Marcap", "Stocks", "MarketId"]
    return fdr.StockListing("KRX")[cols].sort_values(by='Marcap', ascending=False)


def get_list(market="KRX"):
    """
    지수 구성 종목 리스트를 가져옵니다.

    Parameters:
    -----------
    market : str
        시장 코드 (현재 'KRX'만 지원)

    Returns:
    --------
    pd.DataFrame
        종목 리스트 (시가총액 순 정렬)
    """
    if market == 'KRX':
        df = _get_krx_list()
    else:
        raise ValueError(f"Unsupported market: {market}")
    return df


def _download_single_ticker(ticker, start_date, end_date, data_source=None):
    """
    단일 종목 다운로드 (병렬 처리용 내부 함수)

    Parameters:
    -----------
    ticker : str
        종목 코드
    start_date : str
        시작일 (YYYY-MM-DD)
    end_date : str
        종료일 (YYYY-MM-DD)
    data_source : str, optional
        데이터 소스 (예: 'KRX', 'NAVER', 'YAHOO')

    Returns:
    --------
    tuple
        (ticker, DataFrame) 또는 (ticker, None)
    """
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
    종목들의 가격 데이터를 가져옵니다.

    다운로드 방법은 설정 파일(data.download_method)에서 지정:
    - "batch": 일괄 다운로드 (ticker가 10개 이하일 때도 자동 적용)
    - "parallel": 병렬 다운로드 (ticker가 10개 초과일 때)
    - "sequential": 순차 다운로드

    모든 방법은 'close' 컬럼만 포함하는 DataFrame을 반환합니다.

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
    pd.DataFrame
        날짜를 index로 하고 종목 코드를 column으로 하는 종가 DataFrame
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # 설정 로드
    download_method = get_config('data.price.download_method', "parallel")
    data_source = get_config('data.price.download_source', None)
    max_workers = os.cpu_count() or 4  # fallback to 4

    # 배치 다운로드 모드
    if (download_method == "batch") or (download_method == "parallel" and len(tickers) <= 10):
        print("      배치 다운로드 모드")
        tickers_str = ",".join(tickers)
        df = fdr.DataReader(tickers_str, start_date, end_date)
        if not df.empty:
            df.columns = [col.lower() for col in df.columns]
        # 'close' 컬럼만 추출하여 반환 (다른 모드와 일관성 유지)
        if 'close' in df.columns:
            closeD = df[['close']]
        else:
            closeD = pd.DataFrame()
        return closeD

    price_data = {}

    # 병렬 다운로드 모드
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

    # 순차 다운로드 모드
    else:
        print("      순차 다운로드 모드")
        for ticker in tqdm(tickers, desc="Downloading"):
            try:
                ticker, df = _download_single_ticker(ticker, start_date, end_date, data_source)
                if df:
                    price_data[ticker] = df
            except Exception:
                continue

    # 'close' 컬럼만 추출
    close_series = {
        ticker: df['close']
        for ticker, df in price_data.items()
        if 'close' in df.columns
    }
    closeD = pd.DataFrame(close_series)

    return closeD
