"""주식 데이터 수집 모듈"""
from datetime import datetime
import pandas as pd
import FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import requests
from typing import Optional, Set, Tuple
from core.config import settings
from core.console import smart_progress


# ============================================================
# 내부 상수 (Minor Constants - 변경 가능성 낮음)
# ============================================================

# 네이버 API 내부 코드
_MARKET_KOSPI_CODE = "0"
_MARKET_KOSDAQ_CODE = "1"
_MARKET_KOSPI_NAME = "KOSPI"
_MARKET_KOSDAQ_NAME = "KOSDAQ"

_TRADE_TYPE_KRX = "KRX"
_MARKET_TYPE_ALL = "ALL"
_ORDER_TYPE_MARKET_SUM = "marketSum"
_ORDER_TYPE_STATUS_TAG = "statusTag"
_ORDER_TYPE_TRADE_STOP = "tradeStopYn"
_ORDER_TYPE_MARKET_ALERT = "marketAlertType"

_ALERT_TYPE_CAUTION = "01"
_ALERT_TYPE_WARNING = "02"
_ALERT_TYPE_RISK = "03"

# 표준 컬럼명
_STOCK_LIST_COLUMNS = ["Code", "Name", "Market", "Volume", "Amount", "Marcap", "Stocks"]

# 숫자형 변환 대상 컬럼
_NUMERIC_COLUMNS_BASIC = ['Volume', 'Amount', 'Marcap', 'Stocks']
_NUMERIC_COLUMNS_NAVER = [
    'accQuant', 'accAmount', 'marketSum', 'listedStockCnt',
    'propertyTotal', 'debtTotal', 'sales', 'salesIncreasingRate',
    'operatingProfit', 'operatingProfitIncreasingRate', 'netIncome',
    'eps', 'per', 'pbr', 'roe', 'roa', 'dividend', 'reserveRatio'
]


# ============================================================
# Layer 0: 내부 유틸리티 (Private Utilities)
# ============================================================

def _convert_columns_to_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    DataFrame의 지정된 컬럼들을 숫자형으로 변환합니다 (순수 함수).

    Parameters:
    -----------
    df : pd.DataFrame
        변환할 DataFrame
    columns : list
        숫자형으로 변환할 컬럼 리스트

    Returns:
    --------
    pd.DataFrame
        변환된 DataFrame
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _create_naver_column_mapping() -> dict:
    """
    네이버 API 컬럼명 → 표준 컬럼명 매핑 생성 (순수 함수).

    Returns:
    --------
    dict
        컬럼명 매핑 딕셔너리
    """
    return {
        'itemcode': 'Code',
        'itemname': 'Name',
        'accQuant': 'Volume',
        'accAmount': 'Amount',
        'marketSum': 'Marcap',
        'listedStockCnt': 'Stocks',
        'propertyTotal': '자산총계',
        'debtTotal': '부채총계',
        'sales': '매출액',
        'salesIncreasingRate': '매출액증가율',
        'operatingProfit': '영업이익',
        'operatingProfitIncreasingRate': '영업이익증가율',
        'netIncome': '당기순이익',
        'eps': '주당순이익',
        'dividend': '보통주배당금',
        'per': 'PER',
        'roe': 'ROE',
        'roa': 'ROA',
        'pbr': 'PBR',
        'reserveRatio': '유보율',
        'is_management': '관리종목여부',
        'is_trading_halt': '거래정지여부',
        'is_investment_caution': '투자주의여부',
        'is_investment_warning': '투자경고여부',
        'is_investment_risk': '투자위험여부'
    }


def _get_naver_ordered_columns() -> list:
    """
    네이버 데이터 표준 컬럼 순서 반환 (순수 함수).

    Returns:
    --------
    list
        정렬된 컬럼명 리스트
    """
    return [
        'Code', 'Name', 'Market', 'Volume', 'Amount', 'Marcap', 'Stocks',
        '자산총계', '부채총계',
        '매출액', '매출액증가율', '영업이익', '영업이익증가율', '당기순이익',
        '주당순이익', '보통주배당금',
        'PER', 'ROE', 'ROA', 'PBR', '유보율',
        '관리종목여부', '거래정지여부', '투자주의여부', '투자경고여부', '투자위험여부'
    ]


# ============================================================
# Layer 1: API 호출 (Private API Calls)
# ============================================================

def _fetch_naver_api(
    order_type: str = _ORDER_TYPE_MARKET_SUM,
    start_idx: int = 0,
    alert_type: Optional[str] = None
) -> pd.DataFrame:
    """
    네이버 금융 API 호출 (내부 함수).

    Parameters:
    -----------
    order_type : str
        정렬 기준
    start_idx : int
        시작 인덱스
    alert_type : str, optional
        경고 타입

    Returns:
    --------
    pd.DataFrame
        API 응답 데이터

    Raises:
    -------
    RuntimeError
        API 호출 실패 시
    """
    params = {
        "tradeType": _TRADE_TYPE_KRX,
        "marketType": _MARKET_TYPE_ALL,
        "orderType": order_type,
        "startIdx": start_idx,
        "pageSize": settings.fetcher.naver.page_size
    }

    if alert_type:
        params["alertType"] = alert_type

    headers = {
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "referer": settings.fetcher.naver.api_referer,
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    response = requests.get(
        settings.fetcher.naver.api_url,
        params=params,
        headers=headers,
        timeout=settings.fetcher.naver.timeout
    )
    response.raise_for_status()

    data = response.json()

    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict) and "result" in data and "stocks" in data["result"]:
        return pd.DataFrame(data["result"]["stocks"])
    else:
        raise RuntimeError("Unexpected API response format")


def _fetch_krx_api() -> pd.DataFrame:
    """
    KRX API 호출 (FinanceDataReader 사용, 내부 함수).

    Returns:
    --------
    pd.DataFrame
        KRX 종목 리스트

    Raises:
    -------
    RuntimeError
        API 호출 실패 시
    """
    cols = ["Code", "ISU_CD", "Name", "Market", "Volume", "Amount", "Marcap", "Stocks", "MarketId"]
    df = fdr.StockListing("KRX")[cols]
    return df


def _download_single_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    data_source: Optional[str]
) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    단일 종목 가격 데이터 다운로드 (내부 함수, 병렬 처리용).

    Parameters:
    -----------
    ticker : str
        종목 코드
    start_date : str
        시작일 (YYYY-MM-DD)
    end_date : str
        종료일 (YYYY-MM-DD)
    data_source : str, optional
        데이터 소스

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


# ============================================================
# Layer 2: 데이터 변환 (Private Data Transformation)
# ============================================================

def _transform_krx_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    KRX 데이터를 표준 형식으로 변환 (순수 함수).

    Parameters:
    -----------
    df : pd.DataFrame
        KRX API 원본 데이터

    Returns:
    --------
    pd.DataFrame
        표준 형식으로 변환된 데이터
    """
    df = _convert_columns_to_numeric(df, _NUMERIC_COLUMNS_BASIC)
    df = df[_STOCK_LIST_COLUMNS].sort_values(by='Marcap', ascending=False, ignore_index=True)
    return df


def _fetch_special_stocks(order_type: str, alert_type: Optional[str] = None) -> Set[str]:
    """
    특수 종목 코드 집합 가져오기 (내부 함수).

    Parameters:
    -----------
    order_type : str
        정렬 기준
    alert_type : str, optional
        경고 타입

    Returns:
    --------
    set
        종목 코드 집합
    """
    try:
        df = _fetch_naver_api(order_type=order_type, alert_type=alert_type)
        if 'itemcode' in df.columns:
            return set(df['itemcode'].tolist())
    except Exception:
        pass
    return set()


def _add_special_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    네이버 데이터에 특수 종목 플래그 추가 (병렬 호출).

    Parameters:
    -----------
    df : pd.DataFrame
        주식 데이터

    Returns:
    --------
    pd.DataFrame
        특수 종목 플래그가 추가된 DataFrame
    """
    if not settings.fetcher.naver.fetch_special_flags:
        return df

    fetch_params = [
        ('management', _ORDER_TYPE_STATUS_TAG, None),
        ('trading_halt', _ORDER_TYPE_TRADE_STOP, None),
        ('caution', _ORDER_TYPE_MARKET_ALERT, _ALERT_TYPE_CAUTION),
        ('warning', _ORDER_TYPE_MARKET_ALERT, _ALERT_TYPE_WARNING),
        ('risk', _ORDER_TYPE_MARKET_ALERT, _ALERT_TYPE_RISK),
    ]

    special_stocks = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_fetch_special_stocks, order_type, alert_type): name
            for name, order_type, alert_type in fetch_params
        }

        for future in as_completed(futures):
            name = futures[future]
            special_stocks[name] = future.result()

    df['is_management'] = df['itemcode'].isin(special_stocks['management'])
    df['is_trading_halt'] = df['itemcode'].isin(special_stocks['trading_halt'])
    df['is_investment_caution'] = df['itemcode'].isin(special_stocks['caution'])
    df['is_investment_warning'] = df['itemcode'].isin(special_stocks['warning'])
    df['is_investment_risk'] = df['itemcode'].isin(special_stocks['risk'])

    return df


def _transform_naver_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    네이버 API 데이터를 표준 형식으로 변환 (순수 함수).

    Parameters:
    -----------
    df : pd.DataFrame
        네이버 API 원본 데이터

    Returns:
    --------
    pd.DataFrame
        표준 형식으로 변환된 데이터
    """
    # Market 컬럼 생성
    df['Market'] = df['sosok'].astype(str).map({
        _MARKET_KOSPI_CODE: _MARKET_KOSPI_NAME,
        _MARKET_KOSDAQ_CODE: _MARKET_KOSDAQ_NAME
    })

    # 숫자형 변환
    df = _convert_columns_to_numeric(df, _NUMERIC_COLUMNS_NAVER)

    # 컬럼명 변경
    column_mapping = _create_naver_column_mapping()
    df = df.rename(columns=column_mapping)

    # 컬럼 순서 정렬
    ordered_columns = _get_naver_ordered_columns()
    available_columns = [col for col in ordered_columns if col in df.columns]
    df = df[available_columns]

    # Marcap으로 정렬
    if 'Marcap' in df.columns:
        df = df.sort_values(by='Marcap', ascending=False, ignore_index=True)

    return df


# ============================================================
# Layer 3: 다운로드 전략 (Private Download Strategies)
# ============================================================

def _download_batch(
    tickers: list,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    배치 다운로드 방식 (내부 함수).

    Parameters:
    -----------
    tickers : list
        종목 코드 리스트
    start_date : str
        시작일
    end_date : str
        종료일

    Returns:
    --------
    pd.DataFrame
        종가 DataFrame
    """
    tickers_str = ",".join(tickers)
    df = fdr.DataReader(tickers_str, start_date, end_date)

    if not df.empty:
        df.columns = [col.lower() for col in df.columns]

    if 'close' in df.columns:
        return df[['close']]
    else:
        return pd.DataFrame()


def _download_parallel(
    tickers: list,
    start_date: str,
    end_date: str,
    data_source: Optional[str]
) -> pd.DataFrame:
    """
    병렬 다운로드 방식 (내부 함수).

    Parameters:
    -----------
    tickers : list
        종목 코드 리스트
    start_date : str
        시작일
    end_date : str
        종료일
    data_source : str, optional
        데이터 소스

    Returns:
    --------
    pd.DataFrame
        종가 DataFrame
    """
    max_workers = settings.fetcher.price.max_workers or os.cpu_count() or 4
    price_data = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_single_ticker, ticker, start_date, end_date, data_source): ticker
            for ticker in tickers
        }

        for future in smart_progress(as_completed(futures), desc="Downloading", total=len(futures)):
            ticker, df = future.result()
            if df is not None:
                price_data[ticker] = df

    close_series = {
        ticker: df['close']
        for ticker, df in price_data.items()
        if 'close' in df.columns
    }

    return pd.DataFrame(close_series)


def _download_sequential(
    tickers: list,
    start_date: str,
    end_date: str,
    data_source: Optional[str]
) -> pd.DataFrame:
    """
    순차 다운로드 방식 (내부 함수).

    Parameters:
    -----------
    tickers : list
        종목 코드 리스트
    start_date : str
        시작일
    end_date : str
        종료일
    data_source : str, optional
        데이터 소스

    Returns:
    --------
    pd.DataFrame
        종가 DataFrame
    """
    price_data = {}

    for ticker in smart_progress(tickers, desc="Downloading"):
        ticker_code, df = _download_single_ticker(ticker, start_date, end_date, data_source)
        if df is not None:
            price_data[ticker_code] = df

    close_series = {
        ticker: df['close']
        for ticker, df in price_data.items()
        if 'close' in df.columns
    }

    return pd.DataFrame(close_series)


# ============================================================
# Layer 4: 공개 API (Public Interface)
# ============================================================

def get_list(market: str = "KRX", list_source: Optional[str] = None) -> pd.DataFrame:
    """
    지수 구성 종목 리스트를 가져옵니다.

    Parameters:
    -----------
    market : str
        시장 코드 (현재 'KRX'만 지원)
    list_source : str, optional
        데이터 소스 ('KRX' 또는 'Naver'). None이면 설정 파일에서 읽음.

    Returns:
    --------
    pd.DataFrame
        종목 리스트 (시가총액 순 정렬)

    Raises:
    -------
    ValueError
        지원하지 않는 market 또는 list_source
    """
    if market != 'KRX':
        raise ValueError(f"Unsupported market: {market}")

    if list_source is None:
        list_source = settings.stocks.list.source

    if list_source == 'Naver':
        df = _fetch_naver_api()
        df = _add_special_flags(df)
        df = _transform_naver_data(df)
    elif list_source == 'KRX':
        df = _fetch_krx_api()
        df = _transform_krx_data(df)
    else:
        raise ValueError(f"Unsupported list_source: {list_source}")

    return df


def get_price(
    tickers: list,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    종목들의 가격 데이터를 가져옵니다.

    다운로드 방법은 설정 파일(stocks.price.download_method)에서 지정:
    - "batch": 일괄 다운로드
    - "parallel": 병렬 다운로드
    - "sequential": 순차 다운로드

    Parameters:
    -----------
    tickers : list
        종목 코드 리스트
    start_date : str, optional
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

    download_method = settings.stocks.price.download_method
    data_source = settings.stocks.price.download_source
    batch_threshold = settings.fetcher.price.batch_threshold

    # 전략 선택 및 실행
    if download_method == "batch" or (download_method == "parallel" and len(tickers) <= batch_threshold):
        return _download_batch(tickers, start_date, end_date)
    elif download_method == "parallel":
        return _download_parallel(tickers, start_date, end_date, data_source)
    else:  # sequential
        return _download_sequential(tickers, start_date, end_date, data_source)


def get_etf_data(ticker: str, start_date, end_date) -> Optional[pd.Series]:
    """
    ETF 월말 종가 데이터 가져오기.

    Parameters:
    -----------
    ticker : str
        ETF 티커 (예: '069500', '114800')
    start_date : datetime
        시작일
    end_date : datetime
        종료일

    Returns:
    --------
    pd.Series or None
        ETF 월말 종가

    Raises:
    -------
    RuntimeError
        데이터 다운로드 실패 시
    """
    etf_data = fdr.DataReader(ticker, start_date, end_date)
    etf_data.index = pd.to_datetime(etf_data.index)
    etf_monthly = etf_data['Close'].resample('ME').last()
    return etf_monthly
