# KRX300 Quantitative Analysis

한국거래소 시가총액 상위 300 종목(KRX300)에 대한 퀀트 분석 및 대시보드 생성 프로젝트

## 프로젝트 개요

국내 주식 시장 추세 파악을 목적으로 KRX300 종목의 모멘텀, 성과, 상관관계를 분석하고 정적 웹 페이지 형태의 대시보드를 생성합니다.

## 주요 기능

- KRX300 종목 리스트 자동 수집
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
git clone https://github.com/<username>/quant-krx300.git
cd quant-krx300

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

# 결과 확인
open output/index.html
```

## 출력 페이지 구조

생성된 정적 웹사이트는 다음 구조를 가집니다:

```
output/
├── index.html                    # 메인 페이지 (네비게이션)
├── list/
│   ├── universe.html            # KRX300 종목 리스트
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

### STEP 1: KRX300 종목 리스트 생성

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
  n_universe: 300              # KRX300
  price:
    periods: 63                 # 63개월 데이터
    download_method: parallel   # parallel, batch, onebyone

# 시그널 분석 기간
signals:
  momentum:
    periods: [12, 36, 60]       # 12/36/60개월 분석
  correlation:
    periods: 12                  # 12개월 상관관계

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
```

## GitHub Actions 자동화

`.github/workflows/deploy.yml`

매월 1일 오전 10시(KST)에 자동으로 분석을 실행하고 GitHub Pages에 배포합니다.

### 트리거
- 스케줄: 매월 1일 오전 1시(UTC) = 오전 10시(KST)
- 수동 실행: Actions 탭에서 "Run workflow"
- Push: master 브랜치에 push 시

### 배포 URL
`https://<username>.github.io/quant-krx300/`

### 캐싱
- uv 캐시 활성화로 빌드 시간 단축 (5-10분 → 2-3분)
- `uv.lock` 파일 기준으로 캐시 관리

## 프로젝트 구조

```
quant-krx300/
├── .github/
│   └── workflows/
│       ├── deploy.yml          # GitHub Actions 워크플로우
│       └── README.md           # 워크플로우 설명
├── templates/
│   ├── dashboard.html          # 대시보드 템플릿
│   ├── index.html              # 인덱스 페이지 템플릿
│   └── table.html              # 테이블 템플릿
├── output/                      # 생성된 웹사이트 (gitignore)
├── step1_list.py               # 종목 리스트
├── step2_price.py              # 가격 데이터
├── step3_signals.py            # 시그널 계산
├── step4_selector.py           # 종목 선택 및 포트폴리오 구성
├── step5_dashboards.py         # 대시보드 생성
├── step6_index.py              # 인덱스 페이지
├── step_all.py                 # 전체 파이프라인
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

## 향후 계획

### 백테스트 기능

조건:
- Universe: KRX300 종목
- 리밸런싱: 매월 1일, 전월 종가 기준

**전략 1**: 모멘텀 → 품질 → 상관관계 순 필터링
- KRX300 모멘텀 상위 1/2 선택
- (Slope × R-squared) 상위 1/2 선택
- Correlation marginal sum 최소인 상위 1/3 선택
- 모멘텀 > 0 종목 매수, 나머지 현금 보유

**전략 2**: 모멘텀 → 상관관계 → 품질 순 필터링
- KRX300 모멘텀 상위 1/2 선택
- Correlation marginal sum 최소인 상위 1/3 선택
- (Slope × R-squared) 상위 1/2 선택
- 모멘텀 > 0 종목 매수, 나머지 현금 보유

**전략 3**: 클러스터 기반 선택
- KRX300 모멘텀 상위 1/2 선택
- 계층적 클러스터링으로 25개 클러스터 생성
- 각 클러스터에서 (Slope × R-squared) 최상위 1개 선택
- 모멘텀 > 0 종목 매수, 나머지 현금 보유

## 라이선스

MIT License

## 문의

Issues: https://github.com/<username>/quant-krx300/issues
