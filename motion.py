import cv2
import numpy as np
import requests
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# ============================
# TELEGRAM CONFIG (loaded securely from .env — never hardcoded)
# ============================
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID. Create a .env file (see .env.example).")

def send_telegram(image_path):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(image_path, "rb") as img:
        requests.post(url, data={"chat_id": CHAT_ID}, files={"photo": img})

# ============================
# YOLO LOAD (PERSON ONLY)
# ============================
BASE_DIR = Path(__file__).resolve().parent

cfg_path = str(BASE_DIR / "yolov4-tiny.cfg")
weights_path = str(BASE_DIR / "yolov4-tiny.weights")

print("CFG exists:", Path(cfg_path).exists())
print("Weights exists:", Path(weights_path).exists())

net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

# ============================
# LOAD CLASS NAMES
# ============================
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

print("Classes loaded:", classes[:5])

# ============================
# YOLO OUTPUT LAYERS
# ============================
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# ============================
# SETUP
# ============================
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
save_dir = Path("person_alerts")
save_dir.mkdir(exist_ok=True)

last_alert_time = 0

print("Smart Motion + Person Alert Started (Telegram Enabled)")

# ============================
# MAIN LOOP
# ============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    clean_frame = frame.copy()

    fgmask = fgbg.apply(frame)
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion = any(cv2.contourArea(c) > 6000 for c in contours)

    if motion:
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True)
        net.setInput(blob)
        detections = net.forward(output_layers)

        for output in detections:
            for det in output:
                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if classes[class_id] == "person" and confidence > 0.5:
                    h, w = frame.shape[:2]
                    cx, cy, bw, bh = (det[0:4] * np.array([w, h, w, h])).astype(int)

                    x1 = max(cx - bw // 2, 0)
                    y1 = max(cy - bh // 2, 0)
                    x2 = min(cx + bw // 2, w)
                    y2 = min(cy + bh // 2, h)

                    person_img = clean_frame[y1:y2, x1:x2]

                    if person_img.size > 0:
                        filename = save_dir / f"person_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(str(filename), person_img)
                        send_telegram(filename)
                        print("Person detected → Image sent to Telegram")
                        break

    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
