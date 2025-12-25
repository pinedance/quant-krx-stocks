"""
STEP 1: KRX300 종목 리스트 생성
- KRX API에서 KRX300 종목 리스트 가져오기
- HTML, TSV, JSON 형식으로 출력
"""

import json
from datetime import datetime
from core.io import get_list
from jinja2 import Environment, FileSystemLoader

# 템플릿 파일이 있는 디렉토리 설정
file_loader = FileSystemLoader('templates')
env = Environment(loader=file_loader)
template = env.get_template('list.html')

def export_to_html(df, filepath):
    render_data = {
        "title": "KRX300 종목 리스트",
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "dataframe": df.to_html(index=False, escape=False)
    }
    html_content = template.render(render_data)

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
    df = get_list('KRX300')

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
