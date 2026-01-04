# GitHub Actions Workflows

## Deploy to GitHub Pages

이 워크플로우는 KRX 분석을 자동으로 실행하고 결과를 GitHub Pages에 배포합니다.

### 설정 방법

1. **GitHub Pages 활성화**
   - Repository Settings → Pages
   - Source: "GitHub Actions" 선택

2. **워크플로우 실행**
   - 자동 실행: 매월 1일 오전 10시 (KST)
   - 수동 실행: Actions 탭 → "Deploy KRX Analysis to GitHub Pages" → "Run workflow"
   - Push 실행: `main` 브랜치에 push 시

3. **결과 확인**
   - URL: `https://<username>.github.io/<repository-name>/`
   - 예: `https://pinedance.github.io/quant-krx-stocks/`

### 워크플로우 구조

```yaml
jobs:
  build:
    - Checkout repository
    - Install uv
    - Set up Python with uv
    - Install dependencies (uv sync)
    - Run analysis (uv run python step_all.py)
    - Create index.html
    - Upload artifact

  deploy:
    - Deploy to GitHub Pages
```

### 포함된 페이지

- **메인 페이지** (`index.html`): 모든 대시보드와 데이터 링크
- **대시보드**:
  - Momentum Dashboard
  - Performance Dashboard
  - Correlation Network
  - Correlation Cluster
- **시그널 데이터**:
  - Momentum Signals
  - Performance Signals
  - Correlation Matrix
- **종목 리스트**:
  - Universe List

### 주의사항

- 분석 실행 시간: 약 5-10분
- GitHub Actions 무료 사용량: 2,000분/월 (Public repository)
- 데이터는 FinanceDataReader를 통해 실시간으로 수집됨

### 트러블슈팅

#### 워크플로우 실패 시
1. Actions 탭에서 로그 확인
2. 데이터 수집 오류: 외부 API 문제일 수 있음 (재실행)
3. Python 패키지 오류: `uv sync` 확인

#### GitHub Pages 배포 안 됨
1. Repository Settings → Pages에서 Source 확인
2. Permissions 확인 (Settings → Actions → General → Workflow permissions)
3. "Read and write permissions" 활성화

### 수동 실행 방법

GitHub Actions 탭에서:
1. "Deploy KRX Analysis to GitHub Pages" 선택
2. "Run workflow" 버튼 클릭
3. 브랜치 선택 (기본: main)
4. "Run workflow" 실행

### 로컬에서 테스트

```bash
# 전체 파이프라인 실행
uv run python step_all.py

# 개별 스텝 실행
uv run python step1_get_universe.py
uv run python step2_get_price.py
uv run python step3_calc_signals.py
uv run python step4_dashboards.py

# 결과 확인
open output/index.html
```
