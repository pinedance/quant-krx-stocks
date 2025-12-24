# KRX300 프로젝트 사용 가이드

## 개요

KRX300 종목에 대한 모멘텀, 성과, 상관관계 분석 및 대시보드 생성 프로젝트

## 빠른 시작

### 1. 전체 실행 (STEP 1~3 모두)

```bash
uv run python run_all.py
```

### 2. 단계별 실행

```bash
# STEP 1: KRX300 종목 리스트 생성
uv run python step1_list.py

# STEP 2: 데이터 생성 및 저장 (HTML, TSV, JSON)
uv run python step2_data.py

# STEP 3: 대시보드 생성
uv run python step3_dashboards.py
```

## 출력 파일 구조

```
output/
├── list/                          # STEP 1 출력
│   ├── krx300_list.html          # Google Sheets IMPORTHTML용
│   ├── krx300_list.tsv
│   └── krx300_list.json
│
├── data/                          # STEP 2 출력
│   ├── priceD.{html,tsv,json}    # 일간 가격 데이터
│   ├── priceM.{html,tsv,json}    # 월간 가격 데이터
│   ├── momentum.{html,tsv,json}  # 모멘텀 지표
│   ├── performance.{html,tsv,json}  # 성과 지표
│   └── correlation.{html,tsv,json}  # 상관관계 행렬
│
└── dashboard/                     # STEP 3 출력
    ├── momentum.html             # 모멘텀 대시보드
    ├── performance.html          # 성과 대시보드
    ├── correlation_network.html  # VOSviewer 안내 페이지
    ├── correlation_network.json  # VOSviewer 데이터 파일
    └── correlation_cluster.html  # 클러스터링 덴드로그램
```

## 데이터 설명

### STEP 2 지표

#### Momentum 지표 (momentum.{html,tsv,json})
- `1MR ~ 12MR`: 1~12개월 수익률
- `13612MR`: (1MR + 3MR + 6MR + 12MR) / 4
- `AS12, AS36, AS60`: 12, 36, 60개월 Annualized Slope (연율화 기울기)
- `RS12, RS36, RS60`: R-squared (결정계수)

#### Performance 지표 (performance.{html,tsv,json})
- `AR12, AR36, AR60`: Annualized Return (연율화 수익률)
- `SD12, SD36, SD60`: Standard Deviation (표준편차, Sharpe Ratio용)
- `DD12, DD36, DD60`: Downside Deviation (하방편차, Sortino Ratio용)

#### Correlation (correlation.{html,tsv,json})
- 종목 간 상관계수 행렬 (최근 12개월 기준)
- `marginal_sum`: 각 종목의 다른 종목들과의 상관계수 합계

### STEP 3 대시보드

#### 1. Momentum Dashboard (momentum.html)
- **Chart 1**: Monthly Momentum Scatter
  - X축: 1MR (1개월 모멘텀)
  - Y축: 13612MR (평균 모멘텀)
  - 사분면 구분선 표시

- **Chart 2**: Regression Momentum Scatter
  - X축: R-squared
  - Y축: Annualized Slope
  - 12, 36, 60개월을 색깔로 구분
  - 범례 클릭으로 표시/숨김 가능

#### 2. Performance Dashboard (performance.html)
- **Chart 1**: Sharpe Ratio
  - X축: Standard Deviation
  - Y축: Annualized Return
  - 12, 36, 60개월을 색깔로 구분

- **Chart 2**: Sortino Ratio
  - X축: Downside Deviation
  - Y축: Annualized Return
  - 12, 36, 60개월을 색깔로 구분

#### 3. Correlation Network (VOSviewer)
- **파일**: `correlation_network.json`
- **열기 방법**:
  1. [VOSviewer Online](https://app.vosviewer.com/) 접속
  2. "Open file" 클릭
  3. `correlation_network.json` 업로드

- **네트워크 구조**:
  - Node: 각 종목
  - Edge: 상관계수 > 0.5
  - Node 크기: marginal_sum에 비례
  - 색깔: 자동 감지된 클러스터

#### 4. Correlation Cluster (correlation_cluster.html)
- Hierarchical Clustering Dendrogram
- 거리: 1 - correlation
- Method: Average linkage

## Google Sheets 연동

### HTML Table 가져오기

```
=IMPORTHTML("file_url", "table", 1)
```

예시:
```
=IMPORTHTML("https://your-site.com/output/list/krx300_list.html", "table", 1)
```

### TSV 파일 가져오기

Google Sheets에 직접 업로드하거나 Google Drive에 올린 후 연결

### JSON 데이터 활용

Apps Script나 다른 도구로 JSON 파싱 후 활용

## 커스터마이징

### step2_data.py 수정 사항

**종목 수 조정** (line 61):
```python
price_data = get_price(tickers[:50], start_date=start_date)  # 50개만
# 전체 실행:
price_data = get_price(tickers, start_date=start_date)  # 전체 300개
```

**기간 조정** (line 58):
```python
start_date = date_before(years=6)  # 6년
# 더 긴 기간:
start_date = date_before(years=10)  # 10년
```

### step3_dashboards.py 수정 사항

**Correlation threshold 조정** (line 178):
```python
threshold = 0.5  # 현재 설정
# 더 많은 엣지:
threshold = 0.3  # 낮춤
# 더 적은 엣지:
threshold = 0.7  # 높임
```

## 트러블슈팅

### 데이터 다운로드 오류
- 일부 종목 코드가 잘못되었거나 상장폐지된 경우 발생
- 로그에서 "Error fetching" 메시지 확인
- 정상 작동하며, 유효한 종목만 처리됨

### 메모리 부족
- 전체 300개 종목 실행 시 메모리 사용량 증가
- step2_data.py에서 종목 수 줄이기:
  ```python
  price_data = get_price(tickers[:100], ...)
  ```

### Dendrogram이 너무 큼
- 종목 수가 많으면 파일 크기 증가
- `correlation_cluster.html`의 height 조정 필요

## 성능 최적화

### 캐싱
현재는 매번 다시 다운로드합니다. 캐싱을 추가하려면:
```python
# core/io.py에 캐시 로직 추가
import pickle
# 가격 데이터를 pickle로 저장/로드
```

### 병렬 처리
많은 종목 다운로드 시 병렬 처리로 속도 향상:
```python
from concurrent.futures import ThreadPoolExecutor
# get_price()에서 ThreadPoolExecutor 사용
```

## 다음 단계 (STEP 4)

README.md의 STEP 4 백테스트 구현 예정:
- 전략 1, 2, 3 구현
- 월 1회 리밸런싱
- 성과 비교 대시보드

## 라이선스 및 주의사항

- 데이터 출처: KRX, Yahoo Finance (via FinanceDataReader)
- 투자 권유 목적이 아닌 분석 도구입니다
- 실제 투자 시 본인 책임하에 진행하세요
