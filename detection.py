from ultralytics import YOLO
from PIL import Image
from phue import Bridge
from dotenv import load_dotenv
import time
import cv2
import os

# Load a Model
model = YOLO("yolo11n.pt")

IP = os.getenv('IP')

# Connects to bridge
b = Bridge(IP)
b.connect()

# Gets time
time_now = time.strftime('%H:%M')

def set_light_on():

    command = {'on' : True , 'bri' : 254}

    if time_now >= '10:00':
        command = {'on' : True , 'bri' : 254, 'hue' : 41432}
    
    elif time_now >= '20:00' or time_now >= '00:00':
        command = {'on' : True , 'bri' : 77, 'hue' : 8417}

    b.set_light('Office Light', command)

def set_light_off():
    b.set_light('Office Light', 'on', False)

cap = cv2.VideoCapture(0)

detection_status = False

while cap.isOpened:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(frame)

     # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLO11 Tracking", annotated_frame)

    person_detected = False
    for box in results[0].boxes:
        if box.cls == 0:
            person_detected = True
            break

    if person_detected and not detection_status:
        set_light_on()
        detection_status = True

    elif not person_detected and detection_status:
        set_light_off()
        detection_status = False

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()