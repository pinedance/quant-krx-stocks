# KRX Quantitative Analysis

한국거래소 시가총액 상위 300 종목에 대한 퀀트 분석 및 대시보드 생성 프로젝트

## 프로젝트 개요

국내 주식 시장 추세 파악을 목적으로 KRX 종목의 모멘텀, 성과, 상관관계를 분석하고 정적 웹 페이지 형태의 대시보드를 생성합니다.

## 주요 기능

- KRX 종목 리스트 자동 수집
- 가격 데이터 다운로드 (일별/월별)
- 모멘텀 지표 계산 (월간 수익률, 회귀 모멘텀)
- 성과 지표 계산 (Sharpe Ratio, Sortino Ratio)
- 상관관계 분석 (네트워크, 클러스터링)
- 인터랙티브 대시보드 생성 (Plotly)
- GitHub Pages 자동 배포

## 설치 및 실행

### 필요 사항

- Python 3.11 이상
- [uv](https://github.com/astral-sh/uv) (Python 패키지 관리자)

### 설치

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 클론
git clone https://github.com/<username>/quant-krx-stocks.git
cd quant-krx-stocks

# Python 설치 및 의존성 설치
uv python install
uv sync
```

### 실행

```bash
# 전체 파이프라인 실행
uv run python step_all.py

# 개별 스텝 실행
uv run python step1_list.py      # 종목 리스트 생성
uv run python step2_price.py     # 가격 데이터 다운로드
uv run python step3_signals.py   # 시그널 계산
uv run python step4_selector.py  # 종목 선택 및 포트폴리오 구성
uv run python step5_dashboards.py # 대시보드 생성
uv run python step6_index.py     # 인덱스 페이지 생성

# 백테스트 실행
uv run python backtest01.py      # 모멘텀 전략 백테스트 (Strategy 1-6)

# 결과 확인
open output/index.html
```

## 출력 페이지 구조

생성된 정적 웹사이트는 다음 구조를 가집니다:

```
output/
├── index.html                    # 메인 페이지 (네비게이션)
├── list/
│   ├── universe.html            # KRX 종목 리스트
│   ├── universe.tsv
│   └── universe.json
├── price/
│   ├── closeD.html              # 일별 가격 데이터
│   ├── closeD.tsv
│   ├── closeD.json
│   ├── closeM.html              # 월별 가격 데이터
│   ├── closeM.tsv
│   └── closeM.json
├── signal/
│   ├── momentum.html            # 모멘텀 시그널
│   ├── momentum.tsv
│   ├── momentum.json
│   ├── performance.html         # 성과 시그널
│   ├── performance.tsv
│   ├── performance.json
│   ├── correlation.html         # 상관관계 매트릭스
│   ├── correlation.tsv
│   └── correlation.json
├── portfolios/
│   ├── scores.html              # 종목별 점수
│   ├── selected.html            # 선택된 종목
│   └── portfolio.html           # 포트폴리오 구성
├── backtests/
│   └── backtest01/              # 모멘텀 전략 백테스트
│       ├── equity_curves.html   # 자산 곡선
│       ├── monthly_returns.html # 월별 수익률
│       ├── metrics.html         # 성과 지표
│       └── trades.html          # 거래 내역
└── dashboard/
    ├── momentum.html            # 모멘텀 대시보드
    ├── performance.html         # 성과 대시보드
    ├── correlation_network.html # 상관관계 네트워크
    ├── correlation_network.json
    ├── correlation_cluster.html # 계층적 클러스터링
    └── correlation_cluster.json
```

### 대시보드 내용

**모멘텀 대시보드** (`dashboard/momentum.html`)
- Monthly Momentum: R-squared vs Average Momentum
- Regression Momentum: R-squared vs Annualized Slope (12/36/60개월)

**성과 대시보드** (`dashboard/performance.html`)
- Risk-Return Analysis: Standard Deviation vs Annualized Return (Sharpe Ratio)
- Downside Risk-Return Analysis: Downside Deviation vs Annualized Return (Sortino Ratio)

**상관관계 대시보드**
- Network Graph: VOSviewer를 이용한 종목 간 상관관계 네트워크
- Dendrogram: 계층적 클러스터링 (25개 클러스터)

## 분석 파이프라인

### STEP 1: KRX 종목 리스트 생성

`step1_list.py`

- KRX 데이터포털에서 시가총액 상위 300종목 수집
- 종목 코드, 종목명, 시가총액 정보 저장
- 출력: `output/list/universe.*`

### STEP 2: 가격 데이터 수집

`step2_price.py`

- FinanceDataReader를 이용한 가격 데이터 다운로드
- 일별 데이터(closeD): 시가, 고가, 저가, 종가, 거래량 (63개월)
- 월별 데이터(closeM): 일별 데이터를 월별로 리샘플링
- 병렬 다운로드 지원 (설정 가능)
- 출력: `output/price/closeD.*`, `output/price/closeM.*`

### STEP 3: 시그널 계산

`step3_signals.py`

**모멘텀 시그널** (`signal/momentum.*`)
- 1~12MR: 1~12개월 모멘텀
- 13612MR: (1MR+3MR+6MR+12MR)/4
- AS12/36/60: Annualized Slope (12/36/60개월)
- RS12/36/60: R-Squared (12/36/60개월)

**성과 시그널** (`signal/performance.*`)
- AR12/36/60: Annualized Return
- SD12/36/60: Standard Deviation (Sharpe Ratio용)
- DD12/36/60: Downside Deviation (Sortino Ratio용)

**상관관계** (`signal/correlation.*`)
- 12개월 가격 변화 기준 종목 간 상관계수 매트릭스
- Marginal mean 추가 (각 종목의 평균 상관관계)

### STEP 4: 종목 선택 및 포트폴리오 구성

`step4_selector.py`

- Momentum/Performance 지표 기반 복합 점수 계산
- 선택 기준에 따른 종목 필터링 및 순위 매기기
- 포트폴리오 가중치 계산 (equal weight, score weighted 등)
- 출력: `output/signal/scores.*`, `output/signal/selected.*`, `output/signal/portfolio.*`

### STEP 5: 대시보드 생성

`step5_dashboards.py`

- Plotly를 이용한 인터랙티브 차트 생성
- 다중 기간 분석 (12/36/60개월)
- Legend grouping (scatter + trendline 동시 표시/숨김)
- VOSviewer JSON 형식 네트워크 데이터 생성
- 계층적 클러스터링 (ward 방법, 25개 클러스터)
- 출력: `output/dashboard/*`

### STEP 6: 인덱스 페이지 생성

`step6_index.py`

- 모든 대시보드와 데이터 파일로의 네비게이션 페이지 생성
- output 디렉토리 스캔 후 자동 링크 생성
- 출력: `output/index.html`

## 설정 파일

`settings.yaml`

```yaml
# 데이터 수집 설정
data:
  market: 'KRX'
  n_universe: 300              # KRX
  n_buffer: 100                # 데이터 미달 종목을 염두한 버퍼
  price:
    periods: 63                 # 63개월 데이터 (5년 + 버퍼 3개월)
    min_periods: 12             # 최소 충족 데이터 수 (monthly)
    download_source: null       # null=자동, "KRX", "NAVER", "YAHOO"
    download_method: parallel   # parallel, batch, onebyone

# 시그널 분석 기간
signals:
  momentum:
    periods: [12, 36, 60]       # 12/36/60개월 분석
  correlation:
    periods: 12                  # 12개월 상관관계

# 백테스트 설정
backtest:
  end_date: 2025-01-31          # 백테스트 종료일 (null이면 데이터 끝까지)

# 시각화 설정
visualization:
  scatter_plot:
    periods: [12, 36, 60]
    height: 600
    width: 800
    colors:
      - "#FF6B6B"               # 12개월
      - "#4ECDC4"               # 36개월
      - "royalblue"             # 60개월

  correlation_network:
    threshold: 0.5              # 상관계수 threshold

  dendrogram:
    method: ward
    n_cluster: 25
    height_per_item: 15         # Dendrogram 항목당 높이

# 프로젝트 설정
project:
  name: "KRX Quantitative Analysis"
  url: "https://pinedance.github.io/quant-krx-stocks/"
```

## GitHub Actions 자동화

`.github/workflows/deploy.yml`

매월 1일 오전 10시(KST)에 자동으로 분석을 실행하고 GitHub Pages에 배포합니다.

### 트리거
- 스케줄: 매월 1일 오전 1시(UTC) = 오전 10시(KST)
- 수동 실행: Actions 탭에서 "Run workflow"
- Push: master 브랜치에 push 시

### 배포 URL
`https://pinedance.github.io/quant-krx-stocks/`

### 캐싱
- uv 캐시 활성화로 빌드 시간 단축 (5-10분 → 2-3분)
- `uv.lock` 파일 기준으로 캐시 관리

## 프로젝트 구조

```
quant-krx-stocks/
├── .github/
│   └── workflows/
│       ├── deploy.yml          # GitHub Actions 워크플로우
│       └── README.md           # 워크플로우 설명
├── templates/
│   ├── backtest_report.html    # 백테스트 리포트 템플릿
│   ├── correlation_cluster.html # 클러스터 덴드로그램 템플릿
│   ├── correlation_network.html # 네트워크 그래프 템플릿
│   ├── dashboard.html          # 대시보드 템플릿
│   ├── dataframe.html          # 데이터프레임 템플릿
│   ├── datatable.html          # 데이터 테이블 템플릿
│   └── index.html              # 인덱스 페이지 템플릿
├── output/                      # 생성된 웹사이트 (gitignore)
├── core/                        # 코어 모듈
│   ├── backtest.py             # 백테스트 엔진
│   ├── config.py               # 설정 관리
│   ├── fetcher.py              # 데이터 수집
│   ├── file.py                 # 파일 입출력
│   ├── finance.py              # 금융 계산
│   ├── message.py              # 알림 메시지
│   ├── models.py               # 데이터 모델
│   ├── renderer.py             # HTML 렌더링
│   ├── utils.py                # 유틸리티
│   └── visualization.py        # 시각화
├── step1_list.py               # 종목 리스트
├── step2_price.py              # 가격 데이터
├── step3_signals.py            # 시그널 계산
├── step4_selector.py           # 종목 선택 및 포트폴리오 구성
├── step5_dashboards.py         # 대시보드 생성
├── step6_index.py              # 인덱스 페이지
├── step_all.py                 # 전체 파이프라인
├── backtest01.py               # 모멘텀 전략 백테스트 (Strategy 1-6)
├── settings.yaml               # 설정 파일
├── pyproject.toml              # 프로젝트 의존성
└── README.md
```

## 의존성

주요 패키지:
- `finance-datareader`: 한국 주식 데이터 다운로드
- `pandas`, `numpy`: 데이터 처리
- `scipy`, `scikit-learn`: 통계 분석, 클러스터링
- `plotly`: 인터랙티브 차트
- `networkx`: 네트워크 분석
- `jinja2`: HTML 템플릿
- `pyyaml`, `python-box`: 설정 파일 관리

전체 의존성은 `pyproject.toml` 참조

## 백테스트

`backtest01.py` - 모멘텀 기반 전략 백테스트

### 공통 설정
- **대상**: 시총 상위 300개 종목 (조건: 1년 데이터 존재)
- **리밸런싱**: 매월 1일 (가격: 전월 종가)
- **벤치마크**: 069500 (KODEX 200)
- **인버스 ETF**: 114800 (KODEX 인버스)

### 지표 정의
- **13612MR**: (1MR + 3MR + 6MR + 12MR) / 4 → 복합 모멘텀
- **mean-R²**: (1 + √RS3 + √RS6 + √RS12) / 4 → 추세 품질
- **correlation marginal mean**: 상관계수 행렬의 각 종목 평균값 → 분산 효과

### 전략 설명

**Strategy 1 (Base)**
- 필터링: 13612MR 상위 1/2 | mean-R² 상위 1/2 | correlation 하위 1/3
- 포지션: 1/N 동일 비중, 13612MR < 0 종목은 현금 보유
- 종목 수: ~25개

**Strategy 2 (Inverse)**
- 필터링: Strategy 1과 동일
- 포지션: 1/N 동일 비중, 13612MR < 0 종목은 1/4 인버스 + 3/4 현금
- 종목 수: ~25개

**Strategy 3 ★ BEST**
- 필터링: 13612MR 상위 1/2 | mean-R² 상위 1/2 | correlation 하위 1/4
- 포지션: 1/N 동일 비중, 13612MR < 0 종목은 현금 보유
- 종목 수: ~19개
- 특징: 분산 효과 강화 (correlation 1/4)

**Strategy 4 ★ WORST**
- 필터링: 13612MR 상위 1/3 | mean-R² 상위 1/3 | correlation 하위 1/3
- 포지션: 1/N 동일 비중, 13612MR < 0 종목은 현금 보유
- 종목 수: ~11개
- 특징: 모든 필터를 엄격하게 적용 (1/3)

**Strategy 5**
- 필터링: 13612MR 상위 1/2 | mean-R² 상위 1/3 | correlation 하위 1/3
- 포지션: 1/N 동일 비중, 13612MR < 0 종목은 현금 보유
- 종목 수: ~16개
- 특징: 추세 품질 필터 강화

**Strategy 6**
- 필터링: 13612MR 상위 1/2 | mean-R² 상위 1/3 | correlation 하위 1/4
- 포지션: 1/N 동일 비중, 13612MR < 0 종목은 현금 보유
- 종목 수: ~12개
- 특징: 추세 품질 + 분산 효과 강화

### 출력 리포트
- **equity_curves.html**: 자산 곡선 비교
- **monthly_returns.html**: 월별 수익률 히트맵
- **metrics.html**: 성과 지표 (CAGR, MDD, Sharpe, Sortino 등)
- **trades.html**: 거래 내역 및 포지션 변화

## 라이선스

MIT License

## 문의

Issues: https://github.com/pinedance/quant-krx-stocks/issues
