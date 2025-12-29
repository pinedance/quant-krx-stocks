"""
STEP 1: KRX300 종목 리스트 생성
- KRX API에서 KRX300 종목 리스트 가져오기
- HTML, TSV, JSON 형식으로 출력
"""

import json
from datetime import datetime
from core.io import get_list, get_template
from core.config import get_config

def export_to_html(df, template, filepath):
    render_data = {
        "title": "KRX top300 종목 리스트",
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
    print("STEP 1: 종목 리스트 생성")
    print("=" * 60)

    # KRX300 종목 리스트 가져오기
    market = get_config(data.market)

    print("\n종목 리스트를 가져오는 중...")
    df = get_list(market)

    if df.empty:
        print("종목 리스트를 가져오는데 실패했습니다.")
        return

    print(f"총 {len(df)}개 종목을 가져왔습니다.")
    print("\n첫 5개 종목:")
    print(df.head())

    # 3가지 형식으로 저장
    print("\n출력 파일 생성 중...")

    list_dir = get_config(output.list_dir)
    template = get_template(get_config(template.base_dir), 'list.html')
    export_to_html(df, template, f'{list_dir}/{market}_list.html')
    export_to_tsv(df, f'{list_dir}/{market}_list.tsv')
    export_to_json(df, f'{list_dir}/{market}_list.json')

    print("\n" + "=" * 60)
    print("STEP 1 완료!")
    print("=" * 60)
    print("\n생성된 파일:")
    print(f'- {list_dir}/{market}_list.html')
    print(f'- {list_dir}/{market}_list.tsv')
    print(f'- {list_dir}/{market}_list.json')

if __name__ == "__main__":
    main()
