import mediapipe as mp
import cv2
import threading
import time
import numpy as np

class camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.ret, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while self.started:
            ret,frame=self.cap.read()
            with self.read_lock:
                self.ret=ret
                if ret:
                    self.frame=frame

    def read(self):
        with self.read_lock:
            if not self.ret or self.frame is None:
                return self.ret, None
            return self.ret,self.frame.copy()
        
    def stop(self):
        self.started=False
        self.cap.release()

model = 'C:/Users/leaht/Desktop/NICO/code/vision/pose_landmarker_lite.task'


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
latest_result = None
result_lock = threading.Lock() #hold skeleton connec tions

def on_result(result, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    with result_lock:
        latest_result = result
    print(f"Detected {len(result.pose_landmarks)} pose(s) at {timestamp_ms} ms")

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),   # face
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),                            # shoulders + arms
    (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20),  # hands
    (11, 23), (12, 24), (23, 24),                                                # torso
    (23, 25), (25, 27), (24, 26), (26, 28),                                      # legs
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),                  # feet
]

def draw(frame,result):
    h,w,_=frame.shape
    for pose in result.pose_landmarks:
        pts = [(int(lm.x*w), int(lm.y*h)) for lm in pose]
        for a,b in POSE_CONNECTIONS:
            cv2.line(frame, pts[a],pts[b], (0,255,0),2)
        for p in pts:
            cv2.circle(frame,p,4,(0,0,255),-1)

def main():
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_poses=1,
        result_callback=on_result,
    )

    cap = camera(src=0).start()

    with PoseLandmarker.create_from_options(options) as landmarker:
        lastTS = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print('ignoring empty frames')
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            timestamp_ms = max(int(time.time() *1000),lastTS+1)
            lastTS = timestamp_ms

            # Async: automatically drops incoming camera frames when busy so results stay current
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            landmarker.detect_async(mp_image, timestamp_ms)

            with result_lock:
                result = latest_result
            if result is not None:
                draw(frame,result)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



