from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


def date_before(years=0, months=0, days=0, date_format='%Y-%m-%d'):
    """
    현재 날짜 기준으로 과거 날짜를 계산합니다.
    
    Parameters:
    -----------
    years : int
        년수
    months : int
        개월수
    days : int
        일수
    date_format : str
        반환할 날짜 형식
    
    Returns:
    --------
    str
        지정된 형식의 날짜 문자열
    """
    today = datetime.now()
    past_date = today - relativedelta(years=years, months=months, days=days)
    return past_date.strftime(date_format)
