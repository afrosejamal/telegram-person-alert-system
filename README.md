# 📡 Telegram Person Alert System

### *Motion catches your eye. A person triggers your phone.*

A lightweight surveillance script that watches a webcam feed, filters out noise using motion detection, and — only when a **person** is actually confirmed in frame — sends an instant photo alert straight to Telegram. No cloud dashboard, no subscription, just your own bot and your own chat.

📸 *Demo screenshots/video coming soon*

---

## 🧠 The Core Idea

```
  Motion detected? ──► Run YOLOv4-tiny ──► Person found (conf > 50%)? ──► 📤 Send to Telegram
                                        └─► Anything else? ──► Ignore
```

Running a detector on every single frame is wasteful. This script only triggers YOLO inference **when motion is already present** — saving CPU and keeping it responsive enough to run on modest hardware.

---

## ✨ Key Features

- 🚶 **Person-only filtering** — ignores pets, objects, and background clutter; only alerts on confirmed people
- ⚡ **Motion-gated inference** — YOLO only runs when background subtraction first flags movement, not every frame
- 📲 **Instant Telegram alerts** — sends a cropped photo of the detected person directly to your phone via the Telegram Bot API
- 💾 **Local archival** — every alert is also saved locally to a timestamped `person_alerts/` folder
- 🪶 **Lightweight model** — uses YOLOv4-tiny for fast inference without needing a GPU

---

## 🛠️ Built With

`OpenCV` (DNN module + background subtraction) · `YOLOv4-tiny` (Darknet) · `Telegram Bot API` · `python-dotenv`

---

## 🚀 Setup & Run

### 1. Clone and install
```bash
git clone https://github.com/afrosejamal/telegram-person-alert-system.git
cd telegram-person-alert-system
pip install -r requirements.txt
```

### 2. Download YOLOv4-tiny model files
This repo doesn't include the model weights (large binary file). Download these into the project root:
- `yolov4-tiny.weights` and `yolov4-tiny.cfg` — from the [official Darknet/YOLO repo](https://github.com/AlexeyAB/darknet)
- `coco.names` — the standard 80-class COCO label file (included with most YOLO setups)

### 3. Create your Telegram bot
1. Message **[@BotFather](https://t.me/BotFather)** on Telegram → `/newbot` → follow the prompts → copy your bot token
2. Message **[@userinfobot](https://t.me/userinfobot)** to get your numeric chat ID
3. Send your new bot at least one message first (Telegram requires this before it can message you back)

### 4. Configure your credentials
Copy `.env.example` to a new file named `.env`, then fill in your real values:
```
TELEGRAM_BOT_TOKEN=your_actual_token
TELEGRAM_CHAT_ID=your_actual_chat_id
```
**Never commit `.env`** — it's already excluded via `.gitignore`.

### 5. Run it
```bash
python motion.py
```
Press `q` to quit.

---

## ⚙️ How It Works

1. Each frame passes through `MOG2` background subtraction; contours above an area threshold count as motion
2. Only on motion, the frame is converted to a blob and run through YOLOv4-tiny
3. Detections are filtered to the `person` class with confidence above 0.5
4. The bounding box is cropped from a clean (overlay-free) copy of the frame
5. The crop is saved locally and posted to Telegram via `sendPhoto`

---

## 🔐 Security Note

This repo uses environment variables (`.env`, loaded via `python-dotenv`) to keep the Telegram bot token and chat ID out of source code. If you fork this project, **never hardcode your own credentials directly into the script** — always use your own `.env` file, kept out of version control.

---

## ⚠️ Limitations

- Single-camera, single-person-class detection only
- YOLOv4-tiny trades some accuracy for speed — may miss partially occluded people
- No alert deduplication — rapid repeated motion can send multiple alerts in quick succession

## 🔮 Roadmap

- [ ] Add per-person alert cooldown (like the centroid-tracker cooldown in my other motion project)
- [ ] Add demo video/screenshots
- [ ] Support multiple Telegram chat IDs (e.g. alert a group)
- [ ] Optional: switch to YOLOv8 for better accuracy if GPU is available

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Afrose Fathima J**
📧 afrosepvt@gmail.com · 🔗 [LinkedIn](http://www.linkedin.com/in/afrose-fathima-jamal-492b57291)

⭐ *Star this repo if you found it useful!*
