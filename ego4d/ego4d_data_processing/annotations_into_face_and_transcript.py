import csv
import ast
from pathlib import Path
from typing import List, Dict, Union, Iterable

# 사용자가 지정해야 할 해상도
W = 1920  # 원하는 너비
H = 1080  # 원하는 높이

# 처리할 video_uids (persons에 해당)
TARGET_VIDEO_UIDS = {
    "30294c41-c90d-438a-af19-c1c74787d06b",
    "566ad4e5-1ce4-4679-9d19-ef63072c848c",
    "9c5b7322-d1cc-4b56-ae9d-85831f28fac1",
    "9ca2dc18-2c57-44cb-8c91-4b8b5c7ca223",
    "a223fcb2-8ffa-4826-bd0c-91027cf1c11e",
    "b3937482-c973-4263-957d-1d5366329dad",
}

def parse_frame_field(frame_val: Union[str, List[int], None]) -> List[int]:
    """
    frame 필드를 리스트[int]로 파싱.
    - 이미 리스트면 그대로
    - 문자열이면 literal_eval 시도 후 리스트화
    - None/빈 값이면 빈 리스트
    """
    if frame_val is None:
        return []
    if isinstance(frame_val, list):
        return [int(x) for x in frame_val]
    s = str(frame_val).strip()
    if not s:
        return []
    # 쉼표로만 구분된 경우도 허용: "5,6,7"
    if s.startswith('[') and s.endswith(']'):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [int(x) for x in parsed]
        except Exception:
            pass
    # 대괄호가 없으면 쉼표 분리 시도
    try:
        return [int(x.strip()) for x in s.split(',') if x.strip()]
    except Exception:
        return []

def load_transcript_rows(transcript_csv_path: Union[str, Path]) -> Iterable[Dict[str, str]]:
    with open(transcript_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

def build_per_video_annotations(
    transcript_csv_path: Union[str, Path],
    w: int = W,
    h: int = H,
    target_video_uids: set = TARGET_VIDEO_UIDS
) -> Dict[str, List[Dict[str, Union[int, str]]]]:
    """
    transcript CSV를 읽어 각 video_uid별로 어노테이션 레코드 생성.
    반환: {video_uid: [records...]}
    record 필드: frame_number, person_id, x1, y1, x2, y2, speaker_id
    """
    per_video: Dict[str, List[Dict[str, Union[int, str]]]] = {}

    for row in load_transcript_rows(transcript_csv_path):
        video_uid = (row.get('conversation_id') or '').strip()
        if not video_uid or video_uid not in target_video_uids:
            continue

        # speaker_id -> person_id
        spk_raw = row.get('speaker_id', '').strip()
        if spk_raw == '':
            continue
        try:
            speaker_id = int(spk_raw)
        except ValueError:
            # speaker_id가 숫자가 아니면 스킵
            continue

        if speaker_id == 0:
            # 규칙: person_id가 0이면 제외
            continue

        frames = parse_frame_field(row.get('frame'))
        if not frames:
            # 프레임이 없으면 생성할 레코드 없음
            continue

        # 레코드 생성: x1=0, y1=0, x2=w, y2=h
        recs = []
        for fr in frames:
            recs.append({
                'frame_number': int(fr),
                'person_id': speaker_id,
                'x1': 0,
                'y1': 0,
                'x2': int(w),
                'y2': int(h),
                'speaker_id': speaker_id,
            })

        if video_uid not in per_video:
            per_video[video_uid] = []
        per_video[video_uid].extend(recs)

    # 프레임 순서대로 정렬
    for vid, items in per_video.items():
        items.sort(key=lambda x: (x['frame_number'], x['person_id']))

    return per_video

def write_video_csvs(
    per_video_records: Dict[str, List[Dict[str, Union[int, str]]]],
    output_dir: Union[str, Path]
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    header = ['frame_number', 'person_id', 'x1', 'y1', 'x2', 'y2', 'speaker_id']

    for video_uid, records in per_video_records.items():
        out_path = output_dir / f"{video_uid}.csv"
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in records:
                writer.writerow({
                    'frame_number': r['frame_number'],
                    'person_id': r['person_id'],
                    'x1': r['x1'],
                    'y1': r['y1'],
                    'x2': r['x2'],
                    'y2': r['y2'],
                    'speaker_id': r['speaker_id'],
                })

def main(
    transcript_csv_path: Union[str, Path],
    output_dir: Union[str, Path],
    w: int = W,
    h: int = H,
    target_video_uids: set = TARGET_VIDEO_UIDS
):
    per_video = build_per_video_annotations(
        transcript_csv_path=transcript_csv_path,
        w=w,
        h=h,
        target_video_uids=target_video_uids
    )
    write_video_csvs(per_video, output_dir)
    print(f"Done. Wrote {len(per_video)} video files to {output_dir}")

if __name__ == "__main__":
    main(
    # 사용 예시:
    # python script.py 처럼 직접 실행하지 않는다면,
    # 아래 main 호출 부분을 주석 처리하고 외부에서 호출해도 됩니다.
    # main("path/to/transcript.csv", "out_dir", w=1920, h=1080)
    pass
