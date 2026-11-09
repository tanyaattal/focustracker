import cv2
import mediapipe as mp
from pathlib import Path
import time 

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

with FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0) # default camera

    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video capture.")

    try: 
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                break

            timestamp_ms = int(time.perf_counter() * 1000)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            if not result.face_landmarks:
                print("No face landmarks detected.")

            else:
                height, width, _ = frame.shape
                landmarks = result.face_landmarks[0]
                for landmark in landmarks:
                    x_pixel = int(landmark.x * width)
                    y_pixel = int(landmark.y * height)
                    cv2.circle(frame, (x_pixel, y_pixel), 3, (0, 255, 0), -1)

            cv2.imshow("Face Landmarks", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

