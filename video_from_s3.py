import cv2
from ultralytics import YOLO
import time
import pandas as pd
import os
import smtplib
import boto3
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ---------- AWS S3 Configuration ----------
s3_bucket = "your-s3-bucket-name"
s3_video_key = "your-video.mp4"  # S3 key for the video file
local_video_path = "downloaded_video.mp4"

# Initialize S3 client
s3 = boto3.client('s3')

def download_video_from_s3():
    try:
        s3.download_file(s3_bucket, s3_video_key, local_video_path)
        print(f"✅ Video downloaded from S3: {s3_video_key}")
    except Exception as e:
        print(f"❌ Failed to download video: {e}")
        exit(1)

# ---------- Email Settings ----------
from_email = "sprav@gmail.com"
password = ""
to_email = "prv@gmail.com"

# ---------- Create CSV File & Image Directory ----------
csv_file = "alerts_log.csv"
image_folder = "alert_images"

if not os.path.exists(image_folder):
    os.makedirs(image_folder)

if not os.path.exists(csv_file):
    pd.DataFrame(columns=["Timestamp", "Objects Detected", "Image Path"]).to_csv(csv_file, index=False)

# ---------- Function to Send Email Alert ----------
def send_email_alert(image_filename, new_objects):
    msg = MIMEMultipart()
    msg['Subject'] = f"⚠️ Alert: Objects Detected - {', '.join(new_objects)}"
    msg['From'] = from_email
    msg['To'] = to_email

    text = MIMEText(f"New object(s) detected: {', '.join(new_objects)}\nImage saved at: {image_filename}")
    msg.attach(text)

    with open(image_filename, 'rb') as img_file:
        image_data = MIMEImage(img_file.read(), name=os.path.basename(image_filename))
        msg.attach(image_data)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# ---------- Function to Log Alerts (Images + CSV) ----------
def log_alert(image, new_objects):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    image_filename = f"{image_folder}/alert_{int(time.time())}.jpg"

    # Save image
    cv2.imwrite(image_filename, image)

    # Log alert in CSV
    log_data = pd.DataFrame([[timestamp, ", ".join(new_objects), image_filename]], 
                            columns=["Timestamp", "Objects Detected", "Image Path"])
    log_data.to_csv(csv_file, mode='a', header=False, index=False)

    print(f"📂 Alert logged: {timestamp}, Objects: {new_objects}, Image saved: {image_filename}")

    # Send email alert
    send_email_alert(image_filename, new_objects)

# ---------- Function to Process Video ----------
def process_video():
    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        print("❌ Error: Unable to open video.")
        return

    model = YOLO("yolov8n.pt")  # Load YOLOv8 model
    alert_cooldown = 10  # Time in seconds between alerts
    last_alert_time = 0
    baseline_objects = set()
    frame_count = 0
    initialization_frames = 30  # Frames to establish baseline

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        detected_objects = set()

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            detected_objects.add(class_name)

        if frame_count < initialization_frames:
            baseline_objects.update(detected_objects)  # Establish baseline
        else:
            new_objects = detected_objects - baseline_objects
            if new_objects and (time.time() - last_alert_time > alert_cooldown):
                print(f"🚨 Alert: New object(s) detected: {new_objects}")
                log_alert(frame, new_objects)  # Log alert with image
                last_alert_time = time.time()

        cv2.imshow("Monitoring", results[0].plot())  # Display video

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# ---------- Run Script ----------
if __name__ == "__main__":
    download_video_from_s3()
    process_video()
