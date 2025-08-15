import cv2
import os
from dotenv import load_dotenv
from ultralytics import solutions

# Load environment variables
load_dotenv()

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("security_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Get email credentials from environment variables
from_email = os.getenv("FROM_EMAIL")
password = os.getenv("EMAIL_PASSWORD")
to_email = os.getenv("TO_EMAIL")

# Validate that environment variables are loaded
if not all([from_email, password, to_email]):
    raise ValueError("Email credentials not found in environment variables. Check your .env file.")

# Initialize security alarm object
securityalarm = solutions.SecurityAlarm(
    show=True,  # display the output
    model="yolo11n.pt",  # i.e. yolo11s.pt, yolo11m.pt
    records=1,  # total detections count to send an email
)

securityalarm.authenticate(from_email, password, to_email)  # authenticate the email server

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = securityalarm(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows