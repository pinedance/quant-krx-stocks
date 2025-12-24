# 금융 시장 Dash Board

static web site를 만들자. 

국내 시장 추세 파악 목적

## Output Page

KRX300 종목 리스트: data frame 형태
* "https://<URL>/KRX300/list/"

KRX300 data: data frame or matrix 형태
* "https://<URL>/KRX300/data/priceD"
* "https://<URL>/KRX300/data/priceM"
* "https://<URL>/KRX300/data/momentum"
* "https://<URL>/KRX300/data/performance"
* "https://<URL>/KRX300/data/correlation"

KRX300 dashboard: dashboard 형태, plots 중심
* "https://<URL>/KRX300/dashboard/momentum"
* "https://<URL>/KRX300/dashboard/performance"
* "https://<URL>/KRX300/dashboard/correlation/network"
* "https://<URL>/KRX300/dashboard/correlation/cluster"

## STEP 1 : KRX300 종목 리스트 생성

### list 페이지 

web에서 KRX300 종목 리스트 가져오기

Request Header

```
:authority
data.krx.co.kr
:method
POST
:path
/comm/bldAttendant/getJsonData.cmd
:scheme
https
accept
application/json, text/javascript, */*; q=0.01
accept-encoding
gzip, deflate, br, zstd
accept-language
ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7
content-length
219
content-type
application/x-www-form-urlencoded; charset=UTF-8
cookie
__smVisitorID=DAc9VkOr4LC; JSESSIONID=Yoz5dTPOmbXcmjJEMC4n6lLWwpuFPvTLRXv6F3zaqnEOtOWMt11kJqRioL1aqroa.bWRjX2RvbWFpbi9tZGNvd2FwMi1tZGNhcHAwMQ==
dnt
1
origin
https://data.krx.co.kr
priority
u=1, i
referer
https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201010105&idxCd=5&idxCd2=418
sec-ch-ua
"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"
sec-ch-ua-mobile
?0
sec-ch-ua-platform
"Windows"
sec-fetch-dest
empty
sec-fetch-mode
cors
sec-fetch-site
same-origin
user-agent
Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36
x-requested-with
XMLHttpRequest
```

페이로드

```
bld=dbms/MDC/STAT/standard/MDCSTAT00601&locale=ko_KR&tboxindIdx_finder_equidx0_1=KRX+300&indIdx=5&indIdx2=300&codeNmindIdx_finder_equidx0_1=KRX+300&param1indIdx_finder_equidx0_1=&trdDd=20251224&money=3&csvxls_isNo=false
```

## STEP 2: KRX300 data 생성

### priceD & priceM 페이지

priceD & priceM: data frame
* KRX300 price data 다운로드: FinanceDataReader package 사용
* priceD: daily price data (시가, 종가, 고가, 저가), 6년
* priceM: monthly data로 reshape

### momentum 페이지

mmtM: data frame
* 종가 priceM 사용
* row: tickrs
* column: 
    - 1~12MR(1~12 monthly momentum), 13612MR( (1MR+3MR+6MR+12MR)/4 ), 136912MR
    - AS12, RS12, AS36, RS36, AS60, RS60: 12개월, 36개월, 60개월에 대한 Annualized Slope, R-Squared ( log(price)에 대한 선형 회귀 분석 결과 )

### performance 페이지

pfmM: data frame
* 종가 priceM 사용
* row: tickrs
* column: 
    - AR12, AR36, AR60: 12개월, 36개월, 60개월에 대한 Annualized Return
    - SD12, SD36, SD60: 12개월, 36개월, 60개월에 대한 Standard Deviation (for Sharp Ratio)
    - DD12, DD36, DD60: 12개월, 36개월, 60개월에 대한 Downside Deviation (for Sortino Ratio)

### correlation 페이지

corM : Matrix
* 종가 priceM 사용. 12개 데이터(1~12개월)만 사용. 전월 대비 가격 변화를 구한 뒤 상관계수 도출
* ticker 사이의 상관관계
* marginal sum 추가

## STEP 3: KRX300 dashboard 생성

### momentum 페이지

Monthly Momentum: Scatter Plot
* x axis: 1MR
* y axis: 13612MR

Regression Momentum: Scatter Plot
* x axis: R-squared
* y axis: Annualized Slope
* 12개월, 36개월, 60개월을 점의 색깔로 구분. 
    - plot에서 마우스로 색깔별로 감추기 보이기 기능 필요

### performance 페이지

Sharp Ratio: Scatter Plot
* x axis: Standard Deviation
* y axis: Annualized Return
* 12개월, 36개월, 60개월을 점의 색깔로 구분. 
    - plot에서 마우스로 색깔별로 감추기 보이기 기능 필요

Sortino Ratio: Scatter Plot
* x axis: Downside Deviation
* y axis: Annualized Return
* 12개월, 36개월, 60개월을 점의 색깔로 구분. 
    - plot에서 마우스로 색깔별로 감추기 보이기 기능 필요

### correlation/network 페이지

correlation graph: network graph (with VOSviewer)
* node: tickers
* edge weight: correlation
* full screen

### correlation/cluster 페이지

Dendrogram:
* tickers를 상관관계를 기준으로 Hierarchical Clustering


## STEP 4: 벡테스트 기능

조건:
* Universe: KRX300 종목
* price data: monthly close price data
* 월 1일 전월 종가 기준 리벨런싱

전략1:
* KRX300 종목 모멘텀(상대모멘텀) 상위 1/2 선택
* 여기에서 (slop * R-squared) 상위 1/2 선택
* 여기에서 correlation marginal sum이 최소인 상위 1/3 선택
* 여기에서 모멘텀이 0보다 큰 종목에 매수. 나머지 종목 비중 만큼 현금 보유.

전략2:
* KRX300 종목 모멘텀(상대모멘텀) 상위 1/2 선택
* 여기에서 correlation marginal sum이 최소인 상위 1/3 선택
* 여기에서 (slop * R-squared) 상위 1/2 선택
* 여기에서 모멘텀이 0보다 큰 종목에 매수. 나머지 종목 비중 만큼 현금 보유.

전략3:
* KRX300 종목 모멘텀(상대모멘텀) 상위 1/2 선택
* 여기에서 correlation을 기준으로  Hierarchical Clustering 25개 클러스터 도출
* 각 클러스터에서 (slop * R-squared) 상위 1개 선택
* 여기에서 모멘텀이 0보다 큰 종목에 매수. 나머지 종목 비중 만큼 현금 보유.
