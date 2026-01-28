import cv2
import csv
import time
import mediapipe as mp
from pathlib import Path
import time

#采样间隔数据设定
SAMPLE_INTERVAL = 3.0   # 每 3 秒采一次
last_save_time = 0.0
label_change_time = time.time()


# ========== 1) 配置 ==========
OUTPUT_CSV = "pose_dataset.csv"
CAMERA_INDEX = 0          # 如果USB是1，就改成1
SAVE_EVERY_N_FRAMES = 1   # 每N帧保存一次（可设为2/3减少数据量）
SHOW_PREVIEW = True

# 标签映射：你可以自己改
LABEL_MAP = {
    ord('0'): "unknown",
    ord('1'): "good_posture",
    ord('2'): "lean_forward",
    ord('3'): "slouch_back",
    ord('4'): "tilt_left_or_right",
}

# ========== 2) 初始化 MediaPipe Pose ==========
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # 0更快，2更准
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ========== 3) 打开摄像头 ==========
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError(f"❌ 摄像头打开失败（index={CAMERA_INDEX}）")

# ========== 4) 准备 CSV 表头 ==========
# 每个关键点保存 x,y,z,vis，共 33*4 列
landmark_names = [lm.name.lower() for lm in mp_pose.PoseLandmark]
header = ["timestamp", "frame_id", "label", "pose_detected", "avg_visibility"]
for name in landmark_names:
    header += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis"]

csv_path = Path(OUTPUT_CSV)
new_file = not csv_path.exists()

f = open(csv_path, "a", newline="", encoding="utf-8")
writer = csv.writer(f)
if new_file:
    writer.writerow(header)

# ========== 5) 采集循环 ==========
current_label = "unknown"
frame_id = 0

print("✅ 开始采集：按键打标签：0 unknown, 1端正, 2前倾, 3驼背, 4左歪, 5右歪；按 q 退出")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ 读取帧失败")
            break

        frame_id += 1

        # BGR -> RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True

        # 预览画面
        image_bgr = frame
        if SHOW_PREVIEW and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            cv2.putText(
                image_bgr, f"label: {current_label}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        if SHOW_PREVIEW:
            cv2.imshow("Collect Pose Dataset", image_bgr)

        # 键盘输入更新标签
        key = cv2.waitKey(1) & 0xFF
        if key in LABEL_MAP:
            current_label = LABEL_MAP[key]
            print(f"✅ label -> {current_label}")
        elif key == ord('q'):
            break

        # 每N帧保存一次
        key = cv2.waitKey(1) & 0xFF
        if key in LABEL_MAP:
            current_label = LABEL_MAP[key]
            label_change_time = time.time()
            print(f"✅ label -> {current_label}")
        elif key == ord('q'):
            break


        ts = time.time()

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            vis_list = [p.visibility for p in lm]
            avg_vis = sum(vis_list) / len(vis_list)

            row = [ts, frame_id, current_label, 1, avg_vis]
            for p in lm:
                row += [p.x, p.y, p.z, p.visibility]
            writer.writerow(row)
        else:
            # 没检测到人体，也记录一行（对训练也有用：负样本/缺失情况）
            row = [ts, frame_id, current_label, 0, 0.0]
            # 用空值占位（也可以用 0.0）
            for _ in range(33):
                row += ["", "", "", ""]
            writer.writerow(row)

finally:
    f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ 数据已保存到: {csv_path.resolve()}")
