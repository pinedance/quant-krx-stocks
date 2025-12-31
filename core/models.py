"""통계 모델"""
import numpy as np
import pandas as pd


class LM:
    """Linear Model for regression analysis (optimized with vectorization)"""

    def __init__(self):
        self.slope = None
        self.intercept = None
        self.score = None  # R-squared

    def fit(self, data, periods):
        """
        선형 회귀 모델을 학습합니다 (vectorized).

        Parameters:
        -----------
        data : pd.DataFrame or pd.Series
            가격 데이터 (log transformed)
        periods : int
            회귀 기간

        Returns:
        --------
        self
            학습된 모델
        """
        if isinstance(data, pd.DataFrame):
            # DataFrame: 각 컬럼별 회귀 분석 (vectorized)
            slopes = []
            intercepts = []
            scores = []

            # X는 모든 컬럼에 공통
            x = np.arange(periods)
            x_mean = (periods - 1) / 2

            for col in data.columns:
                y = data[col].iloc[-periods:].values

                # 데이터 검증 (선형 회귀는 최소 2개 포인트 필요)
                if periods < 2 or len(y) < periods or not np.all(np.isfinite(y)):
                    slopes.append(np.nan)
                    intercepts.append(np.nan)
                    scores.append(np.nan)
                    continue

                # 직접 계산으로 속도 향상
                y_mean = np.mean(y)

                # Slope 계산: cov(x,y) / var(x)
                numerator = np.sum((x - x_mean) * (y - y_mean))
                denominator = np.sum((x - x_mean) ** 2)
                slope = numerator / denominator

                # Intercept 계산
                intercept = y_mean - slope * x_mean

                # R-squared 계산
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y_mean) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                slopes.append(slope)
                intercepts.append(intercept)
                scores.append(r2)

            # pandas Series로 반환
            self.slope = pd.Series(slopes, index=data.columns)
            self.intercept = pd.Series(intercepts, index=data.columns)
            self.score = pd.Series(scores, index=data.columns)

        elif isinstance(data, pd.Series):
            # Series: 단일 회귀 분석
            y = data.iloc[-periods:].values

            # 데이터 검증 (선형 회귀는 최소 2개 포인트 필요)
            if periods < 2 or len(y) < periods or not np.all(np.isfinite(y)):
                self.slope = np.nan
                self.intercept = np.nan
                self.score = np.nan
                return self

            x = np.arange(periods)
            x_mean = (periods - 1) / 2
            y_mean = np.mean(y)

            # Slope
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            self.slope = numerator / denominator

            # Intercept
            self.intercept = y_mean - self.slope * x_mean

            # R-squared
            y_pred = self.slope * x + self.intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            self.score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        else:
            raise ValueError("data must be DataFrame or Series")

        return self

    def predict(self, x):
        """예측 수행"""
        if isinstance(self.slope, pd.Series):
            # DataFrame 학습 결과
            raise NotImplementedError("Prediction for DataFrame not implemented")
        else:
            # Series 학습 결과
            return self.slope * x + self.intercept
