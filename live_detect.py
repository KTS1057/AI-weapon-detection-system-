import cv2
import time
import smtplib
import ssl
import os
from email.message import EmailMessage
from ultralytics import YOLO

# ---------------- MODELS ----------------
weapon_model = YOLO("best.pt")
person_model = YOLO("yolov8n.pt")

# ---------------- SETTINGS ----------------
CONF_THRESHOLD = 0.45      # lower for knife detection
EMAIL_COOLDOWN = 30        # seconds

SENDER_EMAIL = "2k23.aids2310190@gmail.com"
SENDER_PASSWORD = "iiri ihec arrh nayp"  # set in terminal
RECEIVER_EMAIL = "2k23.aids2310190@gmail.com"

last_email_time = 0

# ---------------- EMAIL FUNCTION ----------------
def send_email(image_path):
    msg = EmailMessage()
    msg["Subject"] = "🚨 WEAPON DETECTED ALERT"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.set_content("Weapon detected with person. Screenshot attached.")

    with open(image_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="image",
            subtype="jpeg",
            filename="weapon.jpg"
        )

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)

# ---------------- OVERLAP FUNCTION ----------------
def is_overlapping(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    weapon_results = weapon_model(frame, imgsz=320, verbose=False)
    person_results = person_model(frame, imgsz=320, verbose=False)

    person_boxes = []
    weapon_boxes = []

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
                cv2.putText(frame, f"Person {conf:.2f}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0),
                            2)

    # -------- WEAPON DETECTION --------
    for result in weapon_results:
        for box in result.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = weapon_model.names[cls]

            if conf > CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                weapon_boxes.append((x1, y1, x2, y2))

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,0,255),
                            2)

    
    # -------- SMART CENTER CHECK --------
weapon_detected = False

for wx1, wy1, wx2, wy2 in weapon_boxes:
    weapon_center_x = (wx1 + wx2) // 2
    weapon_center_y = (wy1 + wy2) // 2

    for px1, py1, px2, py2 in person_boxes:
        if px1 < weapon_center_x < px2 and py1 < weapon_center_y < py2:
            weapon_detected = True
    # -------- ALERT + EMAIL --------
    if weapon_detected:
        cv2.putText(frame, "🚨 WEAPON DETECTED!",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3)

        if time.time() - last_email_time > EMAIL_COOLDOWN:
            screenshot_path = "weapon_detected.jpg"
            cv2.imwrite(screenshot_path, frame)
            send_email(screenshot_path)
            last_email_time = time.time()

    cv2.imshow("Weapon Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
