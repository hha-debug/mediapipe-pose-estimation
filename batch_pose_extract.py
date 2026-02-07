import os
import csv
import json
import cv2
import numpy as np
import mediapipe as mp
import faulthandler

faulthandler.enable()


# ===============================
# 标签说明（你人工标注）
# ===============================
LABEL_MAP = {
    0: "自然坐（不刻意摆）",
    1: "身体前倾+头前伸（贴屏幕看）",
    2: "靠椅背+腰塌+肩向后",
    3: "身体偏一侧（左右均可）"
}

# ===============================
# 旋转角度说明（顺时针）
# 0/90/180/270
# ===============================
ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,  # 等价于逆时针 90
}


def rotate_frame(frame, deg_clockwise: int):
    code = ROTATIONS.get(deg_clockwise)
    if code is None:
        return frame
    return cv2.rotate(frame, code)


# ===============================
# 稳定器：visibility 门控 + EMA 平滑
# ===============================
class LandmarkStabilizer:
    """
    cur: (33,4) -> [x,y,z,vis]  (x,y归一化，z相对)
    规则：
      - vis < vis_th: 不更新（沿用上一帧）
      - vis >= vis_th: 用 EMA 平滑更新
    """
    def __init__(self, alpha=0.2, vis_th=0.5):
        self.alpha = float(alpha)     # 越小越稳(更慢)，越大越跟手(更抖)
        self.vis_th = float(vis_th)   # 越高越严格(更多点冻结)
        self.prev = None              # (33,4)

    def update(self, cur: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev = cur.copy()
            return cur

        out = self.prev.copy()
        vis = cur[:, 3]
        good = vis >= self.vis_th

        # EMA 平滑可靠点
        out[good, :3] = (1.0 - self.alpha) * self.prev[good, :3] + self.alpha * cur[good, :3]
        out[good, 3] = cur[good, 3]

        self.prev = out
        return out


# ===============================
# 主处理函数：单视频离线处理
# ===============================
def process_single_video(
    video_path: str,
    label: int,
    rotation_deg: int,
    out_dir: str = "recording_test",
    merged_csv: str = "recording_test/all_landmarks.csv",
    # 稳定相关参数
    model_complexity: int = 2,
    min_det: float = 0.7,
    min_track: float = 0.7,
    ema_alpha: float = 0.2,
    vis_th: float = 0.5,
    # 额外输出
    save_world_landmarks: bool = True,  # CSV 里额外写 world_x/y/z（更适合DL）
    fill_missing_with_prev: bool = True # 没检测到人体时：用上一帧（更稳），否则写0
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ 找不到视频文件: {video_path}")

    if label not in LABEL_MAP:
        raise ValueError("❌ label 必须是 0/1/2/3")

    if rotation_deg not in ROTATIONS:
        raise ValueError("❌ rotation_deg 必须是 0/90/180/270（顺时针）")

    os.makedirs(out_dir, exist_ok=True)

    video_name = os.path.basename(video_path)
    base = os.path.splitext(video_name)[0]

    out_video_path = os.path.join(out_dir, f"{base}_pose_fix.mp4")
    meta_path = os.path.join(out_dir, f"{base}_meta.json")

    print("\n==============================")
    print("🎬 开始处理视频:", video_name)
    print("标签:", label, "-", LABEL_MAP[label])
    print("旋转修正: 顺时针", rotation_deg, "度")
    print("稳定参数: model_complexity=", model_complexity,
          "min_det=", min_det, "min_track=", min_track,
          "ema_alpha=", ema_alpha, "vis_th=", vis_th)
    print("==============================\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("❌ 无法打开视频")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 30.0

    # 读第一帧，确定旋转后的尺寸
    ret, first = cap.read()
    if not ret:
        raise RuntimeError("❌ 视频为空")

    first = rotate_frame(first, rotation_deg)
    h, w = first.shape[:2]

    # writer 输出“旋转修正 + 标记骨架”的 mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("❌ 输出 mp4 writer 打开失败（可尝试 XVID + .avi）")

    # 统一 CSV 汇总（追加写入）
    csv_exists = os.path.exists(merged_csv)
    csv_f = open(merged_csv, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_f)

    # 表头：包含 image-space landmarks + 可选 world-space
    if not csv_exists:
        header = [
            "video", "label", "frame", "timestamp_ms",
            "rotation_deg_clockwise", "landmark_id",
            "x", "y", "z", "visibility"
        ]
        if save_world_landmarks:
            header += ["world_x", "world_y", "world_z"]
        csv_writer.writerow(header)

    # 初始化 Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=int(model_complexity),
        smooth_landmarks=False,
        min_detection_confidence=float(min_det),
        min_tracking_confidence=float(min_track),
    )

    stabilizer = LandmarkStabilizer(alpha=ema_alpha, vis_th=vis_th)

    # 回到视频开头
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ===============================
    # 进度：尝试读取总帧数（有些视频/解码器可能返回 0）
    # ===============================
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_idx = 0
    missing_frames = 0
    last_good = None  # (33,4) 最后一次稳定后的点

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        frame = rotate_frame(frame, rotation_deg)

        # ===============================
        # ✅ 关键修复：保证输入给 mediapipe 的图像是连续内存 + uint8
        # ===============================
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

        # （可选心跳定位：如果还崩，可以打开）
        # if frame_idx % 50 == 0:
        #     print(f"[hb] before pose.process frame={frame_idx}")

        results = pose.process(rgb)

        # if frame_idx % 50 == 0:
        #     print(f"[hb] after  pose.process frame={frame_idx}")

        # ===============================
        # 进度：每处理 500 帧打印一次
        # ===============================
        if frame_idx > 0 and frame_idx % 500 == 0:
            if total_frames > 0:
                pct = 100.0 * frame_idx / total_frames
                print(f"[进度] {frame_idx}/{total_frames} 帧 ({pct:.2f}%)，missing={missing_frames}")
            else:
                print(f"[进度] 已处理 {frame_idx} 帧，missing={missing_frames}")

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            cur = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)
            cur = stabilizer.update(cur)
            last_good = cur

            # world landmarks（更适合DL，视角变化更鲁棒）
            world = None
            if save_world_landmarks and results.pose_world_landmarks:
                wlm = results.pose_world_landmarks.landmark
                world = np.array([[p.x, p.y, p.z] for p in wlm], dtype=np.float32)  # (33,3)

            # 写 CSV：每帧33行
            for i in range(33):
                row = [
                    video_name, label, frame_idx, float(t_ms),
                    rotation_deg, i,
                    float(cur[i, 0]), float(cur[i, 1]), float(cur[i, 2]), float(cur[i, 3])
                ]
                if save_world_landmarks:
                    if world is None:
                        row += ["", "", ""]
                    else:
                        row += [float(world[i, 0]), float(world[i, 1]), float(world[i, 2])]
                csv_writer.writerow(row)

            # 可视化：用原 results 画骨架（画图用 results 的 landmarks，稳定后的主要用于训练数据）
            vis = frame.copy()
            mp_drawing.draw_landmarks(vis, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(
                vis,
                f"label={label} rot={rotation_deg} vis_th={vis_th} alpha={ema_alpha}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            writer.write(vis)

        else:
            missing_frames += 1

            # 没检测到：更稳的做法是沿用上一帧
            if fill_missing_with_prev and last_good is not None:
                cur = last_good
            else:
                cur = np.zeros((33, 4), dtype=np.float32)

            for i in range(33):
                row = [
                    video_name, label, frame_idx, float(t_ms),
                    rotation_deg, i,
                    float(cur[i, 0]), float(cur[i, 1]), float(cur[i, 2]), float(cur[i, 3])
                ]
                if save_world_landmarks:
                    row += ["", "", ""]
                csv_writer.writerow(row)

            writer.write(frame)

        frame_idx += 1

    cap.release()
    writer.release()
    pose.close()
    csv_f.close()

    meta = {
        "video": video_path,
        "label": label,
        "label_name": LABEL_MAP[label],
        "rotation_deg_clockwise": rotation_deg,
        "fps": float(fps),
        "frames_processed": int(frame_idx),
        "missing_frames": int(missing_frames),
        "stabilizer": {
            "ema_alpha": float(ema_alpha),
            "visibility_threshold": float(vis_th),
            "fill_missing_with_prev": bool(fill_missing_with_prev),
        },
        "pose_model": {
            "model_complexity": int(model_complexity),
            "min_detection_confidence": float(min_det),
            "min_tracking_confidence": float(min_track),
            "save_world_landmarks": bool(save_world_landmarks),
        },
        "outputs": {
            "merged_csv": os.path.abspath(merged_csv),
            "marked_video": os.path.abspath(out_video_path),
        }
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("✅ 完成！")
    print("CSV（追加汇总）:", os.path.abspath(merged_csv))
    print("标记视频输出:", os.path.abspath(out_video_path))
    print("Meta 输出:", os.path.abspath(meta_path))


# ===============================
# 每次一个视频：只改这里
# ===============================
if __name__ == "__main__":
    # 输入视频
    video_path = r"D:\MediaPipe Pose pose_estimation\recordings\07.MP4"

    # 标签（每个视频一种姿势）
    # 0 自然坐
    # 1 身体前倾+头前伸
    # 2 靠椅背+腰塌+肩向后
    # 3 身体偏一侧
    label = 0

    # 旋转修正角度（顺时针）：0/90/180/270
    rotation_deg = 0

    # 稳定参数（斜后方建议更稳一点）
    process_single_video(
        video_path=video_path,
        label=label,
        rotation_deg=rotation_deg,
        out_dir="recording_test",
        merged_csv="recording_test/all_landmarks.csv",
        model_complexity=1,
        min_det=0.7,
        min_track=0.7,
        ema_alpha=0.25,   # 更稳：0.10~0.25可调
        vis_th=0.70,      # 更严格：0.45~0.70可调
        save_world_landmarks=True,
        fill_missing_with_prev=True
    )
