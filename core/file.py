"""DataFrame 입출력 모듈 - 레이어 아키텍처"""
from datetime import datetime
import pandas as pd
import numpy as np
import json
from pathlib import Path
from core.renderer import render_template


# ============================================================
# 설정 (Configuration)
# ============================================================

class ExportConfig:
    """파일 출력 관련 설정"""
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    FLOAT_PRECISION = 2
    SMALL_FLOAT_PRECISION = 6
    SMALL_FLOAT_THRESHOLD = 1.0


# ============================================================
# Layer 0: 유틸리티 함수
# ============================================================

def ensure_directory(path):
    """
    디렉토리가 없으면 생성합니다.

    Parameters:
    -----------
    path : str or Path
        생성할 디렉토리 경로
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def _get_timestamp():
    """현재 시각 문자열 반환"""
    return datetime.now().strftime(ExportConfig.DATE_FORMAT)


def _make_file_path(base_path, extension):
    """파일 경로 생성 (확장자 추가)"""
    return f"{base_path}.{extension}"


def _format_number(value):
    """숫자 포맷팅"""
    if abs(value) < ExportConfig.SMALL_FLOAT_THRESHOLD and value != 0:
        return f"{value:.{ExportConfig.SMALL_FLOAT_PRECISION}f}"
    else:
        return f"{value:.{ExportConfig.FLOAT_PRECISION}f}"


# ============================================================
# Layer 1: 순수 변환 함수 (Pure Functions - 사이드 이펙트 없음)
# ============================================================

def dataframe_to_html_data(df, title):
    """
    DataFrame을 HTML 템플릿 데이터로 변환합니다 (순수 함수).

    Parameters:
    -----------
    df : pd.DataFrame
        변환할 DataFrame
    title : str
        테이블 제목

    Returns:
    --------
    dict
        HTML 템플릿 렌더링용 데이터
    """
    return {
        "title": title,
        "date": _get_timestamp(),
        "n_row": df.shape[0],
        "n_col": df.shape[1],
        "dataframe": df.to_html(index=True, escape=False)
    }


def dataframe_to_datatable_data(df, title):
    """
    DataFrame을 DataTable 템플릿 데이터로 변환합니다 (최적화된 버전).

    Parameters:
    -----------
    df : pd.DataFrame
        변환할 DataFrame
    title : str
        테이블 제목

    Returns:
    --------
    dict
        DataTable 템플릿 렌더링용 데이터
    """
    # 데이터 변환 (vectorized)
    data_rows = []

    for idx in df.index:
        row_values = df.loc[idx].values
        formatted_row = []

        for val in row_values:
            if isinstance(val, (int, float)):
                formatted_row.append(_format_number(val))
            else:
                formatted_row.append(str(val))

        data_rows.append({
            'index': str(idx),
            'row_data': formatted_row
        })

    return {
        "title": title,
        "date": _get_timestamp(),
        "n_row": df.shape[0],
        "n_col": df.shape[1],
        "index_name": df.index.name or 'Index',
        "columns": df.columns.tolist(),
        "data": data_rows
    }


def dataframe_to_json_data(df):
    """
    DataFrame을 JSON 직렬화 가능한 데이터로 변환합니다 (순수 함수).

    Parameters:
    -----------
    df : pd.DataFrame
        변환할 DataFrame

    Returns:
    --------
    dict
        JSON 직렬화 가능한 데이터
    """
    return {
        'created_at': _get_timestamp(),
        'shape': list(df.shape),
        'columns': df.columns.tolist(),
        'index': df.index.tolist() if isinstance(df.index, pd.DatetimeIndex) else df.index.map(str).tolist(),
        'data': df.to_dict(orient='split')['data']
    }


# ============================================================
# Layer 2: 파일 저장 (File I/O)
# ============================================================

def save_file(content, output_path):
    """
    텍스트 콘텐츠를 파일로 저장합니다 (디렉토리 자동 생성).

    Parameters:
    -----------
    content : str
        저장할 텍스트 콘텐츠
    output_path : str
        출력 파일 경로

    Returns:
    --------
    str
        저장된 파일 경로
    """
    ensure_directory(Path(output_path).parent)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return output_path


def save_html(content, output_path):
    """
    HTML 콘텐츠를 파일로 저장합니다 (save_file의 alias).

    Parameters:
    -----------
    content : str
        HTML 콘텐츠 문자열
    output_path : str
        출력 파일 경로

    Returns:
    --------
    str
        저장된 파일 경로
    """
    return save_file(content, output_path)


# ============================================================
# Layer 3: DataFrame Import/Export 함수
# ============================================================

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

    Returns:
    --------
    str
        저장된 파일 경로
    """
    data = dataframe_to_html_data(df, name)
    html_content = render_template('dataframe.html', data)
    html_path = _make_file_path(base_path, 'html')
    return save_file(html_content, html_path)


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

    Returns:
    --------
    str
        저장된 파일 경로
    """
    tsv_path = _make_file_path(base_path, 'tsv')
    ensure_directory(Path(tsv_path).parent)
    df.to_csv(tsv_path, sep='\t', encoding='utf-8', index=include_index)
    return tsv_path


def export_dataframe_to_json(df, base_path):
    """
    DataFrame을 JSON 파일로 저장합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        저장할 DataFrame
    base_path : str
        파일 경로 (확장자 제외)

    Returns:
    --------
    str
        저장된 파일 경로
    """
    data = dataframe_to_json_data(df)
    json_path = _make_file_path(base_path, 'json')
    ensure_directory(Path(json_path).parent)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return json_path


def export_dataframe_to_datatable(df, base_path, name):
    """
    DataFrame을 DataTables 인터랙티브 HTML로 저장합니다.

    Parameters:
    -----------
    df : pd.DataFrame
        저장할 DataFrame
    base_path : str
        파일 경로 (확장자 제외)
    name : str
        테이블 제목

    Returns:
    --------
    str
        저장된 파일 경로
    """
    data = dataframe_to_datatable_data(df, name)
    html_content = render_template('datatable.html', data)
    html_path = _make_file_path(base_path, 'datatable.html')
    return save_file(html_content, html_path)


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

    Returns:
    --------
    dict
        저장된 파일 경로들 {'html': str, 'tsv': str, 'json': str}
    """
    return {
        'html': export_dataframe_to_html(df, base_path, name),
        'tsv': export_dataframe_to_tsv(df, base_path, include_index),
        'json': export_dataframe_to_json(df, base_path)
    }


# ============================================================
# Layer 4: 출력 헬퍼 함수 (호출자에서 사용)
# ============================================================

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

    Returns:
    --------
    dict
        저장된 파일 경로들
    """
    print(f"  {title.lower()}:")
    paths = export_dataframe_to_formats(df, base_path, title)
    for ext, path in paths.items():
        print(f"  ✓ {path}")
    return paths
