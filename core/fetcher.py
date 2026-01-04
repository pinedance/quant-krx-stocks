"""주식 데이터 수집 모듈"""
from datetime import datetime
import pandas as pd
import FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import requests
from typing import Optional, Set
from core.config import get_config


def _get_krx_list():
    """KRX 전체 종목 리스트를 가져옵니다 (FinanceDataReader 사용)"""
    cols = ["Code", "ISU_CD", "Name", "Market", "Volume", "Amount", "Marcap", "Stocks", "MarketId"]
    return fdr.StockListing("KRX")[cols].sort_values(by='Marcap', ascending=False)


def _get_naver_stock_data(
    trade_type: str = "KRX",
    market_type: str = "ALL",
    order_type: str = "marketSum",
    start_idx: int = 0,
    page_size: int = 500,
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
    url = "https://stock.naver.com/api/domestic/market/stock/default"

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
        "referer": "https://stock.naver.com/market/stock/kr/stocklist/capitalization",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
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


def _get_special_stocks(order_type: str, alert_type: Optional[str] = None) -> Set[str]:
    """
    특수 종목 코드 집합을 가져옵니다 (내부 함수).

    Args:
        order_type: 정렬 기준 (statusTag, tradeStopYn, marketAlertType)
        alert_type: 경고 타입

    Returns:
        종목 코드 집합
    """
    df = _get_naver_stock_data(order_type=order_type, alert_type=alert_type)
    if df is not None and 'itemcode' in df.columns:
        return set(df['itemcode'].tolist())
    return set()


def _get_naver_list():
    """네이버 금융 API에서 KRX 종목 리스트를 가져옵니다 (내부 함수)"""
    # 메인 주식 리스트 가져오기
    df = _get_naver_stock_data(page_size=500)

    if df is None or df.empty:
        print("네이버 API에서 데이터를 가져오는데 실패했습니다.")
        return pd.DataFrame()

    # 특수 종목 데이터 가져오기
    management_stocks = _get_special_stocks("statusTag")
    trading_halt_stocks = _get_special_stocks("tradeStopYn")
    investment_caution_stocks = _get_special_stocks("marketAlertType", "01")
    investment_warning_stocks = _get_special_stocks("marketAlertType", "02")
    investment_risk_stocks = _get_special_stocks("marketAlertType", "03")

    # Boolean 컬럼 추가
    df['is_management'] = df['itemcode'].isin(management_stocks)
    df['is_trading_halt'] = df['itemcode'].isin(trading_halt_stocks)
    df['is_investment_caution'] = df['itemcode'].isin(investment_caution_stocks)
    df['is_investment_warning'] = df['itemcode'].isin(investment_warning_stocks)
    df['is_investment_risk'] = df['itemcode'].isin(investment_risk_stocks)

    # 컬럼 이름 매핑 (기존 포맷과 호환)
    column_mapping = {
        'itemcode': 'Code',
        'itemname': 'Name',
        'sosok': 'MarketCode',
        'accQuant': 'Volume',
        'accAmount': 'Amount',
        'marketSum': 'Marcap',
        'listedStockCnt': 'Stocks',
        'nowVal': 'Close',
        'openVal': 'Open',
        'highVal': 'High',
        'lowVal': 'Low',
        'changeVal': 'Change',
        'changeRate': 'ChgRate',
        'eps': 'EPS',
        'per': 'PER',
        'pbr': 'PBR',
        'roe': 'ROE',
        'roa': 'ROA',
        'dividend': 'DPS',
        'dividendRate': 'DividendYield',
        'propertyTotal': 'Assets',
        'debtTotal': 'Liabilities',
        'sales': 'Revenue',
        'operatingProfit': 'OperatingIncome',
        'netIncome': 'NetIncome',
        'frgnRate': 'ForeignRate',
        'reserveRatio': 'ReserveRatio',
        'is_management': 'IsManagement',
        'is_trading_halt': 'IsTradingHalt',
        'is_investment_caution': 'IsInvestmentCaution',
        'is_investment_warning': 'IsInvestmentWarning',
        'is_investment_risk': 'IsInvestmentRisk'
    }

    # 컬럼명 변경
    df = df.rename(columns=column_mapping)

    # Market 컬럼 추가 (MarketCode 0=KOSPI, 1=KOSDAQ)
    if 'MarketCode' in df.columns:
        df['Market'] = df['MarketCode'].map({'0': 'KOSPI', '1': 'KOSDAQ'})

    # Marcap으로 정렬
    if 'Marcap' in df.columns:
        df = df.sort_values(by='Marcap', ascending=False)

    return df


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
        list_source = get_config('stocks.list.source', 'KRX')

    print(f"데이터 소스: {list_source}")

    if list_source == 'Naver':
        df = _get_naver_list()
    elif list_source == 'KRX':
        df = _get_krx_list()
    else:
        raise ValueError(f"Unsupported list_source: {list_source}")

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
    download_method = get_config('stocks.price.download_method', "parallel")
    data_source = get_config('stocks.price.download_source', None)
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
