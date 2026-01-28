import cv2

def find_cameras(max_index=10):
    cams = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Windows 用 DSHOW 更稳
        if not cap.isOpened():
            cap.release()
            continue

        # 读几帧再判断（有些摄像头第一帧空）
        ok = False
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                ok = True
                h, w = frame.shape[:2]
                cams.append((i, w, h))
                break

        cap.release()
        if ok:
            pass
    return cams

cams = find_cameras(10)
if not cams:
    raise RuntimeError("❌ 没找到任何可用摄像头（检查USB、占用、权限）")

print("✅ 检测到可用摄像头：")
for idx, w, h in cams:
    print(f"  index={idx}  frame={w}x{h}")

# 一般 USB 外置不是 0，你也可以手动选
usb_index = None
for idx, _, _ in cams:
    if idx != 0:
        usb_index = idx
        break

if usb_index is None:
    print("⚠️ 只检测到 index=0（可能USB没插好，或被占用）")
    usb_index = 0

print(f"👉 将尝试打开：index={usb_index}")

cap = cv2.VideoCapture(usb_index, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError(f"❌ 打不开摄像头 index={usb_index}")

# 可选：设置分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue
    cv2.imshow(f"Camera index={usb_index}", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
