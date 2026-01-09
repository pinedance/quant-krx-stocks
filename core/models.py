"""
통계 모델 모듈

선형 회귀 분석을 위한 LM(Linear Model) 클래스를 제공합니다.
- DataFrame 전용 API (타입 분기 제거)
- 레이어 아키텍처 (Layer 0: 순수 계산, Layer 1: 공개 API)
- 벡터화된 수치 계산으로 성능 최적화
"""

import numpy as np
import pandas as pd


# ============================================================
# 내부 상수 (Minor Constants)
# ============================================================

_MIN_REGRESSION_POINTS = 2  # 선형 회귀 최소 데이터 포인트
_EPSILON = 1e-10  # 0 검증을 위한 작은 값


# ============================================================
# Layer 0: 내부 계산 함수 (Private)
# ============================================================

def _validate_regression_data(y: np.ndarray, periods: int) -> bool:
    """
    회귀 데이터 유효성 검증 (순수 함수).

    선형 회귀를 수행하기 위해서는:
    - 최소 2개의 데이터 포인트 필요
    - 모든 값이 유한해야 함 (NaN, Inf 불가)

    Parameters:
    -----------
    y : np.ndarray
        회귀할 데이터
    periods : int
        회귀 기간

    Returns:
    --------
    bool
        데이터가 유효하면 True, 아니면 False
    """
    return (
        periods >= _MIN_REGRESSION_POINTS and
        len(y) >= periods and
        np.all(np.isfinite(y))
    )


def _calculate_linear_regression(y: np.ndarray) -> tuple:
    """
    선형 회귀 계산 (순수 함수, 벡터화).

    최소제곱법(OLS)을 사용하여 slope, intercept, R²을 계산합니다.

    수학적 배경:
    - Slope: β = Σ[(x-x̄)(y-ȳ)] / Σ[(x-x̄)²]
    - Intercept: α = ȳ - β·x̄
    - R²: 1 - (SS_res / SS_tot)

    Parameters:
    -----------
    y : np.ndarray
        회귀할 데이터 (1차원)

    Returns:
    --------
    tuple
        (slope, intercept, r_squared)
    """
    periods = len(y)

    # X 축: 0, 1, 2, ..., periods-1
    x = np.arange(periods)
    x_mean = (periods - 1) / 2
    y_mean = np.mean(y)

    # Slope 계산: cov(x,y) / var(x)
    x_centered = x - x_mean
    y_centered = y - y_mean

    numerator = np.sum(x_centered * y_centered)
    denominator = np.sum(x_centered ** 2)
    slope = numerator / denominator

    # Intercept 계산
    intercept = y_mean - slope * x_mean

    # R-squared 계산
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)  # 잔차 제곱합
    ss_tot = np.sum(y_centered ** 2)    # 총 제곱합

    # ss_tot이 0이면 모든 y값이 동일 → R² = 0
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > _EPSILON else 0.0

    return slope, intercept, r_squared


# ============================================================
# Layer 1: 공개 API (Public Interface)
# ============================================================

class LM:
    """
    Linear Model for regression analysis.

    DataFrame의 각 컬럼에 대해 독립적으로 선형 회귀를 수행합니다.
    최소제곱법(OLS)을 사용하여 slope, intercept, R²을 계산합니다.

    Attributes:
    -----------
    slope : pd.Series
        각 종목의 기울기 (log 가격 기준)
    intercept : pd.Series
        각 종목의 절편
    score : pd.Series
        각 종목의 R² (결정계수, 0~1)

    Examples:
    ---------
    >>> model = LM()
    >>> model.fit(log_prices, periods=12)
    >>> annualized_return = np.exp(model.slope * 12) - 1
    """

    def __init__(self):
        self.slope = None
        self.intercept = None
        self.score = None  # R-squared

    def fit(self, data: pd.DataFrame, periods: int):
        """
        선형 회귀 모델을 학습합니다.

        각 컬럼(종목)에 대해 독립적으로 최근 periods 개의 데이터로
        선형 회귀를 수행합니다.

        Parameters:
        -----------
        data : pd.DataFrame
            가격 데이터 (rows=dates, cols=tickers)
            일반적으로 log-transformed 가격 데이터 사용
        periods : int
            회귀 기간 (개월 수)

        Returns:
        --------
        self
            학습된 모델 (메서드 체이닝 지원)
        """
        slopes = []
        intercepts = []
        scores = []

        for col in data.columns:
            y = data[col].iloc[-periods:].values

            if _validate_regression_data(y, periods):
                slope, intercept, r_squared = _calculate_linear_regression(y)
                slopes.append(slope)
                intercepts.append(intercept)
                scores.append(r_squared)
            else:
                slopes.append(np.nan)
                intercepts.append(np.nan)
                scores.append(np.nan)

        self.slope = pd.Series(slopes, index=data.columns)
        self.intercept = pd.Series(intercepts, index=data.columns)
        self.score = pd.Series(scores, index=data.columns)

        return self

    def predict(self, x):
        """
        학습된 모델로 예측을 수행합니다.

        Parameters:
        -----------
        x : int, float, array-like
            예측할 x 값(들)
            - 단일 값: 해당 시점의 예측값 반환
            - 배열: 각 시점의 예측값 반환

        Returns:
        --------
        pd.Series or pd.DataFrame
            예측값
            - x가 스칼라면 pd.Series (각 종목의 예측값)
            - x가 배열이면 pd.DataFrame (rows=x, cols=tickers)

        Examples:
        ---------
        >>> model = LM().fit(log_prices, periods=12)
        >>> # 다음 달(13번째) 예측
        >>> next_month = model.predict(13)
        >>> # 향후 3개월 예측
        >>> future = model.predict([13, 14, 15])
        """
        if np.isscalar(x):
            # 단일 시점 예측
            return self.slope * x + self.intercept
        else:
            # 다중 시점 예측
            x_array = np.array(x).reshape(-1, 1)  # (n_points, 1)
            slope_array = self.slope.values.reshape(1, -1)  # (1, n_tickers)
            intercept_array = self.intercept.values.reshape(1, -1)

            predictions = x_array @ slope_array + intercept_array

            return pd.DataFrame(
                predictions,
                index=x,
                columns=self.slope.index
            )
