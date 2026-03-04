import streamlit as st
import cv2
import time
import smtplib
import ssl
import os
from email.message import EmailMessage
from ultralytics import YOLO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Gun Detection", layout="wide")
st.title("🔫 AI Gun Detection System")
st.markdown("Real-time Gun Detection with Email Alerts")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    weapon_model = YOLO("best.pt")
    person_model = YOLO("yolov8n.pt")
    return weapon_model, person_model

weapon_model, person_model = load_models()

CONF_THRESHOLD = 0.55
EMAIL_COOLDOWN = 30  # seconds

# ---------------- EMAIL SETTINGS ----------------
SENDER_EMAIL = "2k23.aids2310190@gmail.com"
SENDER_PASSWORD = "iiri ihec arrh nayp"
RECEIVER_EMAIL = "2k23.aids2310190@gmail.com"

# ---------------- SESSION STATE ----------------
if "running" not in st.session_state:
    st.session_state.running = False

if "count" not in st.session_state:
    st.session_state.count = 0

if "last_email_time" not in st.session_state:
    st.session_state.last_email_time = 0

# ---------------- EMAIL FUNCTION ----------------
def send_email(image_path):
    msg = EmailMessage()
    msg["Subject"] = "🚨 GUN DETECTED ALERT"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.set_content("Gun detected. Screenshot attached.")

    with open(image_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="image",
            subtype="jpeg",
            filename="gun_detected.jpg"
        )

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)

# ---------------- BUTTONS ----------------
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Detection"):
        st.session_state.running = True

with col2:
    if st.button("⏹ Stop Detection"):
        st.session_state.running = False

status_placeholder = st.empty()
frame_placeholder = st.empty()

# ---------------- DETECTION LOOP ----------------
if st.session_state.running:

    cap = cv2.VideoCapture(0)

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        weapon_results = weapon_model(frame, imgsz=320, verbose=False)
        person_results = person_model(frame, imgsz=320, verbose=False)

        person_boxes = []
        gun_boxes = []

        # -------- PERSON DETECTION --------
        for result in person_results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = person_model.names[cls]

                if label == "person" and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # -------- GUN DETECTION --------
        for result in weapon_results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = weapon_model.names[cls]

                if label == "gun" and conf > CONF_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    gun_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)

        # -------- SMART CENTER CHECK --------
        gun_detected = False

        for gx1, gy1, gx2, gy2 in gun_boxes:
            center_x = (gx1 + gx2) // 2
            center_y = (gy1 + gy2) // 2

            for px1, py1, px2, py2 in person_boxes:
                if px1 < center_x < px2 and py1 < center_y < py2:
                    gun_detected = True

        # -------- ALERT + EMAIL --------
        if gun_detected:
            st.session_state.count += 1
            status_placeholder.error(
                f"🚨 GUN DETECTED! Total Detections: {st.session_state.count}"
            )

            screenshot_path = "gun_detected.jpg"
            cv2.imwrite(screenshot_path, frame)

            current_time = time.time()

            if current_time - st.session_state.last_email_time > EMAIL_COOLDOWN:
                send_email(screenshot_path)
                st.session_state.last_email_time = current_time

        else:
            status_placeholder.success("Status: Safe")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

    cap.release()
