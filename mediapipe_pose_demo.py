import cv2
import mediapipe as mp
import time
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ⭐ Windows 强烈推荐 CAP_DSHOW
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("❌ 摄像头打开失败")

# ========= 录像相关配置 =========
recording = False
writer = None

# 建议先固定输出目录
output_dir = "recordings"
os.makedirs(output_dir, exist_ok=True)

def start_writer(frame_width, frame_height, fps):
    # Windows 常用：mp4v -> .mp4（有些环境更稳的是 XVID -> .avi）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"pose_demo_{ts}.mp4")
    w = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
    if not w.isOpened():
        raise RuntimeError("❌ VideoWriter 打开失败：请尝试把编码改成 XVID 并输出 .avi")
    print(f"🎬 开始录制 -> {out_path}")
    return w

# 读取相机参数（有的摄像头拿不到fps，这里做兜底）
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 1:
    fps = 30  # 兜底
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

while True:
    ret, frame = cap.read()

    # ⭐ 防炸关键
    if not ret or frame is None:
        print("⚠️ 空帧，跳过")
        continue

    # BGR → RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    results = pose.process(image_rgb)

    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        lm = results.pose_landmarks.landmark
        shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        cv2.putText(
            image_bgr,
            f"Shoulder z: {shoulder.z:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # ========= 录制状态提示 =========
    if recording:
        cv2.putText(
            image_bgr,
            "REC",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # ========= 写入视频（保存的是叠加骨架后的画面）=========
    if recording and writer is not None:
        writer.write(image_bgr)

    cv2.imshow("MediaPipe Pose", image_bgr)

    key = cv2.waitKey(1) & 0xFF

    # r：开始/停止录制
    if key == ord('r'):
        recording = not recording
        if recording:
            # 开始录制时再创建 writer（避免空文件）
            writer = start_writer(frame_width, frame_height, fps)
        else:
            # 停止录制释放 writer
            if writer is not None:
                writer.release()
                writer = None
            print("🛑 停止录制")

    # q：退出
    if key == ord('q'):
        break

# ========= 清理资源 =========
if writer is not None:
    writer.release()

cap.release()
cv2.destroyAllWindows()
