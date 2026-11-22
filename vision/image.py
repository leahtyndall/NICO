import mediapipe as mp
import cv2

model = 'C:/Users/leaht/Desktop/NICO/code/vision/model/pose_landmarker_lite.task'
image_path = 'C:/Users/leaht/Desktop/NICO/code/vision/stand.jfif'  # change to your image

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),   # face
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),                            # shoulders + arms
    (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20),  # hands
    (11, 23), (12, 24), (23, 24),                                                # torso
    (23, 25), (25, 27), (24, 26), (26, 28),                                      # legs
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),                  # feet
]

def draw(frame, result):
    h, w, _ = frame.shape
    for pose in result.pose_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose]
        for a, b in POSE_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 0, 255), -1)

def main():
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
    )

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Couldn't load image: {image_path}")
        return

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    with PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    print(f"Detected {len(result.pose_landmarks)} pose(s)")
    draw(frame, result)

    cv2.imwrite('skeleton_output.jpg', frame)  # optional: save it
    cv2.imshow('frame', frame)
    cv2.waitKey(0)  # wait for any key
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()