"""
STEP 6: 결과물 인덱스 페이지 생성
- output 디렉토리의 모든 결과물을 정리한 index.html 생성
"""

from pathlib import Path
from datetime import datetime
from core.config import settings
from core.renderer import render_html_from_template


def scan_directory_files(dir_path, base_path):
    """디렉토리의 파일들을 스캔하여 그룹화

    Parameters:
    -----------
    dir_path : Path
        스캔할 디렉토리
    base_path : Path
        base_dir (상대 경로 계산용)

    Returns:
    --------
    list of dict
        파일 아이템 리스트
    """
    if not dir_path.exists():
        return []

    # 파일들을 base_name으로 그룹화
    file_groups = {}

    for file_path in sorted(dir_path.rglob('*')):
        if file_path.is_file():
            # 상대 경로
            rel_path = file_path.relative_to(base_path)
            parent_name = file_path.parent.name if file_path.parent != dir_path else ''
            file_name = file_path.stem
            file_ext = file_path.suffix[1:]  # .html -> html

            # 그룹 키 생성
            if parent_name:
                group_key = f"{parent_name}/{file_name}"
            else:
                group_key = file_name

            if group_key not in file_groups:
                file_groups[group_key] = {
                    'name': group_key,
                    'description': '',
                    'files': []
                }

            file_groups[group_key]['files'].append({
                'path': str(rel_path),
                'label': file_ext.upper(),
                'type': file_ext
            })

    return list(file_groups.values())


def scan_output_directory(base_dir):
    """output 디렉토리를 settings.output 기반으로 동적 스캔

    Returns:
    --------
    list of dict
        섹션 정보 (title, description, items)
    """
    base_path = Path(base_dir)
    sections = []
    total_files = 0
    total_items = 0

    # settings.output에서 *_dir 필드들을 순서대로 읽음
    section_number = 1
    for key, value in settings.output.items():
        # base_dir, html 등은 스킵
        if key == 'base_dir' or key == 'html' or not key.endswith('_dir'):
            continue

        # dict 형식인 경우만 처리
        if not isinstance(value, dict) or 'path' not in value:
            continue

        dir_path = Path(value['path'])
        title = value.get('title', key.replace('_dir', '').title())
        desc = value.get('desc', '')

        # 해당 디렉토리가 존재하지 않으면 스킵
        if not dir_path.exists():
            continue

        # 파일 스캔
        data_items = scan_directory_files(dir_path, base_path)

        if data_items:
            sections.append({
                'title': f"{section_number}. {title}",
                'description': desc,
                'data_items': data_items
            })
            section_number += 1
            total_items += len(data_items)
            total_files += sum(len(item['files']) for item in data_items)

    return sections, total_files, total_items


def main():
    print("=" * 70)
    print("STEP 6: 결과물 인덱스 페이지 생성")
    print("=" * 70)

    # 설정 로드
    base_dir = settings.output.base_dir
    project_name = settings.project.name

    # 파일 스캔
    print("\n[1/2] 파일 스캔 중...")
    sections, total_files, total_items = scan_output_directory(base_dir)

    print(f"      섹션: {len(sections)}개")
    print(f"      항목: {total_items}개")
    print(f"      파일: {total_files}개")

    # 인덱스 HTML 생성
    print("\n[2/2] 인덱스 페이지 생성 중...")
    render_data = {
        'title': 'KRX Analysis Results',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'project_name': project_name,
        'n_sections': len(sections),
        'n_items': total_items,
        'n_files': total_files,
        'sections': sections
    }

    output_path = f'{base_dir}/index.html'
    render_html_from_template('index.html', render_data, output_path)
    print(f"  ✓ {output_path}")

    print("\n" + "=" * 70)
    print("STEP 6 완료!")
    print(f"브라우저에서 {output_path} 파일을 열어보세요.")
    print("=" * 70)


if __name__ == "__main__":
    main()
