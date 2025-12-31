"""DataFrame 입출력 모듈"""
from datetime import datetime
from pathlib import Path
import pandas as pd
import json
from core.config import get_config
from core.renderer import get_template


def import_dataframe_from_json(json_path):
    """
    JSON 파일에서 DataFrame을 가져옵니다.

    Parameters:
    -----------
    json_path : str
        JSON 파일 경로

    Returns:
    --------
    pd.DataFrame
        로드된 DataFrame
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(
        data['data'],
        index=data['index'],
        columns=data['columns']
    )
    return df


def export_dataframe_to_html(df, base_path, name):
    """
    DataFrame을 HTML 테이블로 저장합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        저장할 DataFrame
    base_path : str
        파일 경로 (확장자 제외)
    name : str
        테이블 제목
    """
    # 출력 디렉토리 확인
    Path(base_path).parent.mkdir(parents=True, exist_ok=True)

    # 템플릿 로드
    template_dir = get_config("template.base_dir")
    template = get_template(template_dir, 'dataframe.html')

    # HTML 렌더링 데이터
    render_data = {
        "title": name,
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "n_row": df.shape[0],
        "n_col": df.shape[1],
        "dataframe": df.to_html(index=True, escape=False)
    }

    html_content = template.render(render_data)
    html_path = f"{base_path}.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"  ✓ {html_path}")


def export_dataframe_to_tsv(df, base_path, include_index=True):
    """
    DataFrame을 TSV 파일로 저장합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        저장할 DataFrame
    base_path : str
        파일 경로 (확장자 제외)
    include_index : bool
        index 포함 여부
    """
    # 출력 디렉토리 확인
    Path(base_path).parent.mkdir(parents=True, exist_ok=True)

    tsv_path = f"{base_path}.tsv"
    df.to_csv(tsv_path, sep='\t', encoding='utf-8', index=include_index)
    print(f"  ✓ {tsv_path}")


def export_dataframe_to_json(df, base_path):
    """
    DataFrame을 JSON 파일로 저장합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        저장할 DataFrame
    base_path : str
        파일 경로 (확장자 제외)
    """
    # 출력 디렉토리 확인
    Path(base_path).parent.mkdir(parents=True, exist_ok=True)

    json_path = f"{base_path}.json"
    data = {
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'shape': list(df.shape),
        'columns': df.columns.tolist(),
        'index': df.index.tolist() if isinstance(df.index, pd.DatetimeIndex) else df.index.map(str).tolist(),
        'data': df.to_dict(orient='split')['data']
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  ✓ {json_path}")


def export_dataframe_to_formats(df, base_path, name, include_index=True):
    """
    DataFrame을 HTML, TSV, JSON 형식으로 일괄 저장합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        저장할 DataFrame
    base_path : str
        파일 경로 (확장자 제외)
    name : str
        테이블 제목
    include_index : bool
        TSV 저장 시 index 포함 여부
    """
    export_dataframe_to_html(df, base_path, name)
    export_dataframe_to_tsv(df, base_path, include_index)
    export_dataframe_to_json(df, base_path)


def export_with_message(df, base_path, title):
    """
    메시지 출력과 함께 DataFrame을 export합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        저장할 DataFrame
    base_path : str
        파일 경로 (확장자 제외)
    title : str
        파일 제목
    """
    print(f"  {title.lower()}:")
    export_dataframe_to_formats(df, base_path, title)
