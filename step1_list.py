"""
STEP 1: KRX300 종목 리스트 생성
- KRX API에서 KRX300 종목 리스트 가져오기
- HTML, TSV, JSON 형식으로 출력
"""

import json
from datetime import datetime
from core.io import get_krx300_list


def export_to_html(df, filepath):
    """DataFrame을 HTML table로 저장 (Google Sheets IMPORTHTML용)"""
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>KRX300 종목 리스트</title>
    <style>
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>KRX300 종목 리스트</h1>
    <p>생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
{df.to_html(index=False, escape=False)}
</body>
</html>"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML 파일 저장: {filepath}")


def export_to_tsv(df, filepath):
    """DataFrame을 TSV(Tab-Separated Values)로 저장"""
    df.to_csv(filepath, sep='\t', index=False, encoding='utf-8')
    print(f"TSV 파일 저장: {filepath}")


def export_to_json(df, filepath):
    """DataFrame을 JSON으로 저장"""
    # records 형식: [{col1: val1, col2: val2}, ...]
    data = {
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'count': len(df),
        'data': df.to_dict(orient='records')
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"JSON 파일 저장: {filepath}")


def main():
    print("=" * 60)
    print("STEP 1: KRX300 종목 리스트 생성")
    print("=" * 60)

    # KRX300 종목 리스트 가져오기
    print("\nKRX300 종목 리스트를 가져오는 중...")
    df = get_krx300_list()

    if df.empty:
        print("종목 리스트를 가져오는데 실패했습니다.")
        return

    print(f"총 {len(df)}개 종목을 가져왔습니다.")
    print("\n첫 5개 종목:")
    print(df.head())

    # 3가지 형식으로 저장
    print("\n출력 파일 생성 중...")

    export_to_html(df, 'output/list/krx300_list.html')
    export_to_tsv(df, 'output/list/krx300_list.tsv')
    export_to_json(df, 'output/list/krx300_list.json')

    print("\n" + "=" * 60)
    print("STEP 1 완료!")
    print("=" * 60)
    print("\n생성된 파일:")
    print("- output/list/krx300_list.html (Google Sheets IMPORTHTML용)")
    print("- output/list/krx300_list.tsv")
    print("- output/list/krx300_list.json")


if __name__ == "__main__":
    main()
