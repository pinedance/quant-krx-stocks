"""주식 데이터 수집 모듈"""
from datetime import datetime
import pandas as pd
import FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import requests
from typing import Optional, Set
from core.config import settings
from core.console import smart_progress

# ============================================================
# Constants
# ============================================================

# Stock list columns (표준 컬럼명)
STOCK_LIST_COLUMN_NAMES = ["Code", "Name", "Market", "Volume", "Amount", "Marcap", "Stocks"]

# ----------------------------------------
# Naver API 관련 상수
# ----------------------------------------

# Naver API 엔드포인트 및 헤더
NAVER_API_URL = "https://stock.naver.com/api/domestic/market/stock/default"
NAVER_API_REFERER = "https://stock.naver.com/market/stock/kr/stocklist/capitalization"
NAVER_API_TIMEOUT = 10

# Naver API 요청 파라미터 기본값
DEFAULT_PAGE_SIZE = 500          # pageSize: 한 번에 가져올 종목 수
DEFAULT_START_INDEX = 0          # startIdx: 시작 인덱스

# Naver API 요청 파라미터: tradeType (거래 타입)
TRADE_TYPE_KRX = "KRX"

# Naver API 요청 파라미터: marketType (시장 타입)
MARKET_TYPE_ALL = "ALL"

# Naver API 요청 파라미터: orderType (정렬 기준)
ORDER_TYPE_MARKET_SUM = "marketSum"     # 시가총액순
ORDER_TYPE_STATUS_TAG = "statusTag"     # 관리종목 필터용
ORDER_TYPE_TRADE_STOP = "tradeStopYn"   # 거래정지 필터용
ORDER_TYPE_MARKET_ALERT = "marketAlertType"  # 투자경고 필터용

# Naver API 요청 파라미터: alertType (투자경고 타입)
ALERT_TYPE_CAUTION = "01"  # 투자주의
ALERT_TYPE_WARNING = "02"  # 투자경고
ALERT_TYPE_RISK = "03"     # 투자위험

# Naver API 응답 데이터: sosok 필드 값 (시장 구분 코드)
MARKET_KOSPI_CODE = "0"    # KOSPI 시장
MARKET_KOSDAQ_CODE = "1"   # KOSDAQ 시장

# ----------------------------------------
# 데이터 변환 관련 상수
# ----------------------------------------

# 시장 이름 매핑 (Naver API sosok -> 표준 Market 컬럼)
MARKET_KOSPI_NAME = "KOSPI"
MARKET_KOSDAQ_NAME = "KOSDAQ"

# 숫자형 변환 대상 컬럼 (KRX/FinanceDataReader 데이터)
NUMERIC_COLUMNS_BASIC = ['Volume', 'Amount', 'Marcap', 'Stocks']

# 숫자형 변환 대상 컬럼 (Naver API 데이터)
NUMERIC_COLUMNS_NAVER = [
    'accQuant', 'accAmount', 'marketSum', 'listedStockCnt',
    'propertyTotal', 'debtTotal', 'sales', 'salesIncreasingRate',
    'operatingProfit', 'operatingProfitIncreasingRate', 'netIncome',
    'eps', 'per', 'pbr', 'roe', 'roa', 'dividend', 'reserveRatio'
]

# ----------------------------------------
# 가격 데이터 다운로드 관련 상수
# ----------------------------------------

# FinanceDataReader 다운로드 방식
DOWNLOAD_METHOD_BATCH = "batch"          # 일괄 다운로드 (콤마로 연결)
DOWNLOAD_METHOD_PARALLEL = "parallel"    # 병렬 다운로드
DOWNLOAD_METHOD_SEQUENTIAL = "sequential"  # 순차 다운로드

# 다운로드 방식 결정 기준
BATCH_THRESHOLD = 10  # ticker 개수가 이 값 이하면 batch 모드 자동 적용

# 병렬 처리 설정
DEFAULT_MAX_WORKERS = 4  # ThreadPoolExecutor 기본 worker 수 (CPU count 사용 불가 시)

# ============================================================
# Helper Functions (공통 유틸리티)
# ============================================================

def _convert_columns_to_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    DataFrame의 지정된 컬럼들을 숫자형으로 변환합니다.

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


# ============================================================
# Stock List Functions - KRX
# ============================================================

def _krx_get_list():
    """KRX 전체 종목 리스트를 가져옵니다 (FinanceDataReader 사용)"""
    cols = ["Code", "ISU_CD", "Name", "Market", "Volume", "Amount", "Marcap", "Stocks", "MarketId"]
    df = fdr.StockListing("KRX")[cols]

    # 숫자형 컬럼 변환 (문자열 -> float)
    df = _convert_columns_to_numeric(df, NUMERIC_COLUMNS_BASIC)

    # STOCK_LIST_COLUMN_NAMES에 맞춰 컬럼 선택 및 정렬
    df = df[STOCK_LIST_COLUMN_NAMES].sort_values(by='Marcap', ascending=False, ignore_index=True)

    return df


# ============================================================
# Stock List Functions - Naver
# ============================================================

def _naver_fetch_stock_data(
    trade_type: str = TRADE_TYPE_KRX,
    market_type: str = MARKET_TYPE_ALL,
    order_type: str = ORDER_TYPE_MARKET_SUM,
    start_idx: int = DEFAULT_START_INDEX,
    page_size: int = DEFAULT_PAGE_SIZE,
    alert_type: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    네이버 금융 API에서 주식 데이터를 가져옵니다 (내부 함수).

    Args:
        trade_type: 거래 타입 (기본값: KRX)
        market_type: 시장 타입 (기본값: ALL)
        order_type: 정렬 기준 (기본값: marketSum - 시가총액순)
        start_idx: 시작 인덱스
        page_size: 페이지 크기
        alert_type: 경고 타입 (01: 투자주의, 02: 투자경고, 03: 투자위험)

    Returns:
        pd.DataFrame 또는 None
    """
    params = {
        "tradeType": trade_type,
        "marketType": market_type,
        "orderType": order_type,
        "startIdx": start_idx,
        "pageSize": page_size
    }

    if alert_type:
        params["alertType"] = alert_type

    headers = {
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "referer": NAVER_API_REFERER,
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(NAVER_API_URL, params=params, headers=headers, timeout=NAVER_API_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict) and "result" in data and "stocks" in data["result"]:
            return pd.DataFrame(data["result"]["stocks"])
        else:
            return None
    except Exception as e:
        print(f"네이버 API 요청 실패: {e}")
        return None


def _naver_fetch_special_stocks(order_type: str, alert_type: Optional[str] = None) -> Set[str]:
    """
    특수 종목 코드 집합을 가져옵니다 (내부 함수).

    Args:
        order_type: 정렬 기준 (statusTag, tradeStopYn, marketAlertType)
        alert_type: 경고 타입

    Returns:
        종목 코드 집합
    """
    df = _naver_fetch_stock_data(order_type=order_type, alert_type=alert_type)
    if df is not None and 'itemcode' in df.columns:
        return set(df['itemcode'].tolist())
    return set()


def _naver_add_special_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    특수 종목 플래그를 DataFrame에 추가합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        주식 데이터

    Returns:
    --------
    pd.DataFrame
        특수 종목 플래그가 추가된 DataFrame
    """
    # 특수 종목 데이터 가져오기
    management_stocks = _naver_fetch_special_stocks(ORDER_TYPE_STATUS_TAG)
    trading_halt_stocks = _naver_fetch_special_stocks(ORDER_TYPE_TRADE_STOP)
    investment_caution_stocks = _naver_fetch_special_stocks(ORDER_TYPE_MARKET_ALERT, ALERT_TYPE_CAUTION)
    investment_warning_stocks = _naver_fetch_special_stocks(ORDER_TYPE_MARKET_ALERT, ALERT_TYPE_WARNING)
    investment_risk_stocks = _naver_fetch_special_stocks(ORDER_TYPE_MARKET_ALERT, ALERT_TYPE_RISK)

    # Boolean 컬럼 추가
    df['is_management'] = df['itemcode'].isin(management_stocks)
    df['is_trading_halt'] = df['itemcode'].isin(trading_halt_stocks)
    df['is_investment_caution'] = df['itemcode'].isin(investment_caution_stocks)
    df['is_investment_warning'] = df['itemcode'].isin(investment_warning_stocks)
    df['is_investment_risk'] = df['itemcode'].isin(investment_risk_stocks)

    return df


def _naver_transform_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    네이버 API 데이터의 컬럼을 표준 형식으로 변환합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        네이버 API에서 가져온 데이터

    Returns:
    --------
    pd.DataFrame
        컬럼이 변환된 DataFrame
    """
    # Market 컬럼 생성 (sosok: 0=KOSPI, 1=KOSDAQ)
    df['Market'] = df['sosok'].astype(str).map({
        MARKET_KOSPI_CODE: MARKET_KOSPI_NAME,
        MARKET_KOSDAQ_CODE: MARKET_KOSDAQ_NAME
    })

    # 숫자형 컬럼 변환 (문자열 -> float)
    df = _convert_columns_to_numeric(df, NUMERIC_COLUMNS_NAVER)

    # 컬럼 이름 매핑 (한글)
    column_mapping = {
        'itemcode': 'Code',
        'itemname': 'Name',
        'accQuant': 'Volume',
        'accAmount': 'Amount',
        'marketSum': 'Marcap',
        'listedStockCnt': 'Stocks',
        # 재무상태표
        'propertyTotal': '자산총계',
        'debtTotal': '부채총계',
        # 손익계산서
        'sales': '매출액',
        'salesIncreasingRate': '매출액증가율',
        'operatingProfit': '영업이익',
        'operatingProfitIncreasingRate': '영업이익증가율',
        'netIncome': '당기순이익',
        # 주당지표
        'eps': '주당순이익',
        'dividend': '보통주배당금',
        # 재무비율
        'per': 'PER',
        'roe': 'ROE',
        'roa': 'ROA',
        'pbr': 'PBR',
        'reserveRatio': '유보율',
        # Boolean 컬럼
        'is_management': '관리종목여부',
        'is_trading_halt': '거래정지여부',
        'is_investment_caution': '투자주의여부',
        'is_investment_warning': '투자경고여부',
        'is_investment_risk': '투자위험여부'
    }

    # 컬럼명 변경
    df = df.rename(columns=column_mapping)

    # 컬럼 순서 정의 (STOCK_LIST_COLUMN_NAMES 기준 + 추가 컬럼)
    ordered_columns = [
        # 기본 컬럼 (STOCK_LIST_COLUMN_NAMES)
        'Code', 'Name', 'Market', 'Volume', 'Amount', 'Marcap', 'Stocks',
        # 재무상태표
        '자산총계', '부채총계',
        # 손익계산서
        '매출액', '매출액증가율', '영업이익', '영업이익증가율', '당기순이익',
        # 주당지표
        '주당순이익', '보통주배당금',
        # 재무비율
        'PER', 'ROE', 'ROA', 'PBR', '유보율',
        # Boolean 컬럼
        '관리종목여부', '거래정지여부', '투자주의여부', '투자경고여부', '투자위험여부'
    ]

    # 존재하는 컬럼만 선택하여 순서대로 배치
    available_columns = [col for col in ordered_columns if col in df.columns]
    df = df[available_columns]

    # Marcap으로 정렬
    if 'Marcap' in df.columns:
        df = df.sort_values(by='Marcap', ascending=False, ignore_index=True)

    return df


def _naver_get_list():
    """네이버 금융 API에서 KRX 종목 리스트를 가져옵니다 (내부 함수)"""
    # 메인 주식 리스트 가져오기
    df = _naver_fetch_stock_data(page_size=DEFAULT_PAGE_SIZE)

    if df is None or df.empty:
        print("네이버 API에서 데이터를 가져오는데 실패했습니다.")
        return pd.DataFrame()

    # 특수 종목 플래그 추가
    df = _naver_add_special_flags(df)

    # 컬럼 변환 및 정렬
    df = _naver_transform_columns(df)

    return df


# ============================================================
# Stock List Functions - Public Interface
# ============================================================

def get_list(market="KRX", list_source=None):
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
    """
    if market != 'KRX':
        raise ValueError(f"Unsupported market: {market}")

    # list_source가 지정되지 않으면 설정에서 읽기
    if list_source is None:
        list_source = settings.stocks.list.source

    print(f"데이터 소스: {list_source}")

    if list_source == 'Naver':
        df = _naver_get_list()
    elif list_source == 'KRX':
        df = _krx_get_list()
    else:
        raise ValueError(f"Unsupported list_source: {list_source}")

    return df


# ============================================================
# Stock Price Functions
# ============================================================

def _price_download_single_ticker(ticker, start_date, end_date, data_source=None):
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
    download_method = settings.stocks.price.download_method
    data_source = settings.stocks.price.download_source
    max_workers = os.cpu_count() or DEFAULT_MAX_WORKERS

    # 배치 다운로드 모드
    if (download_method == DOWNLOAD_METHOD_BATCH) or (download_method == DOWNLOAD_METHOD_PARALLEL and len(tickers) <= BATCH_THRESHOLD):
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
    if download_method == DOWNLOAD_METHOD_PARALLEL and len(tickers) > BATCH_THRESHOLD:
        print(f"      병렬 다운로드 모드 (workers={max_workers})")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_price_download_single_ticker, ticker, start_date, end_date, data_source): ticker
                for ticker in tickers
            }
            for future in smart_progress(as_completed(futures), desc="Downloading", total=len(futures)):
                ticker, df = future.result()
                if df is not None:
                    price_data[ticker] = df

    # 순차 다운로드 모드
    else:
        print("      순차 다운로드 모드")
        for ticker in smart_progress(tickers, desc="Downloading"):
            try:
                ticker, df = _price_download_single_ticker(ticker, start_date, end_date, data_source)
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


# ============================================================
# ETF Data Functions
# ============================================================

def get_etf_data(ticker: str, start_date, end_date) -> Optional[pd.Series]:
    """
    ETF 월말 종가 데이터 가져오기

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
    """
    try:
        # ETF 데이터 다운로드
        etf_data = fdr.DataReader(ticker, start_date, end_date)
        etf_data.index = pd.to_datetime(etf_data.index)

        # 월말 종가만 추출
        etf_monthly = etf_data['Close'].resample('ME').last()

        return etf_monthly
    except Exception as e:
        print(f"  경고: {ticker} 데이터를 가져올 수 없습니다: {e}")
        return None
