import cv2
import os

def get_video_properties(video_path):
    """
    동영상 파일의 총 프레임 수와 FPS를 계산합니다.

    Args:
        video_path (str): 동영상 파일 경로.

    Returns:
        tuple: (총 프레임 수, FPS) 또는 (None, None) (파일을 열 수 없는 경우).
    """
    # 동영상 파일이 실제로 존재하는지 확인
    if not os.path.exists(video_path):
        print(f"오류: '{video_path}' 파일을 찾을 수 없습니다.")
        return None, None

    # 동영상 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)

    # 동영상이 성공적으로 열렸는지 확인
    if not cap.isOpened():
        print(f"오류: '{video_path}' 파일을 열 수 없습니다.")
        return None, None

    # 동영상 속성 가져오기
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 캡처 객체 해제
    cap.release()

    return frame_count, fps

# 동영상 파일 경로 (사용자 환경에 맞게 수정해주세요)
video_path = '/Volumes/T7 Shield Portable/Github/ego-datasets/aria/recording.mp4'

# 함수 호출 및 결과 출력
total_frames, fps = get_video_properties(video_path)

if total_frames is not None:
    print(f"동영상 경로: {video_path}")
    print(f"총 프레임 수: {total_frames} 프레임")
    print(f"초당 프레임 (FPS): {fps:.2f}")