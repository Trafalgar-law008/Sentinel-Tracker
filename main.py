import cv2
import torch
import pygame
import os
from ultralytics import YOLO
from twilio.rest import Client

pygame.mixer.init()
alert_sound = "alert.mp3.wav"
if os.path.exists(alert_sound):
    pygame.mixer.music.load(alert_sound)

animal_model = YOLO("models/animal_model.pt")  # Replace with your trained model path
human_model = YOLO("models/human_model.pt")    # Replace with pre-trained YOLO model path

ANIMAL_CLASSES = {idx: name.upper() for idx, name in animal_model.names.items()}

DANGEROUS_ANIMALS = ["ELEPHANT", "TIGER", "LEOPARD", "LION"]
SAFE_ANIMALS = ["BISON", "DEER"]

TWILIO_ACCOUNT_SID = "ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  
TWILIO_AUTH_TOKEN = "your_auth_token_here"  
TWILIO_PHONE_NUMBER = "+10000000000"  
USER_PHONE_NUMBER = "+911234567890"  

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_sms_alert(animal_name, x, y, w, h):
    message_body = f"⚠ ALERT: {animal_name} detected!\nLocation: (x:{x}, y:{y}, w:{w}, h:{h}). Stay safe!"
    try:
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=USER_PHONE_NUMBER
        )
        print(f" SMS Sent Successfully! Message SID: {message.sid}")
    except Exception as e:
        print(f"SMS Sending Failed: {e}")

def play_sound_alert():
    if os.path.exists(alert_sound):
        try:
            pygame.mixer.music.play()
        except Exception as e:
            print(f" Sound Error: {e}")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(" Camera Not Working. Check Webcam.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Human detection
    human_results = human_model(frame_rgb, conf=0.5, iou=0.5)
    human_detected = False  

    for result in human_results:
        if hasattr(result, "boxes"):
            class_ids = result.boxes.cls.cpu().numpy()
            for class_id in class_ids:
                if int(class_id) == 0: 
                    human_detected = True
                    break  

    if human_detected:
        print(" Ignoring Humans – No False Animal Alerts!")
        cv2.putText(frame, "Human Detected - Ignoring!", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Animal detection
        animal_results = animal_model(frame_rgb, conf=0.4, iou=0.5)

        for result in animal_results:
            if not hasattr(result, "boxes"):
                continue  

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                width, height = x2 - x1, y2 - y1
                confidence = confs[i]
                class_id = int(class_ids[i])

                animal_name = ANIMAL_CLASSES.get(class_id, "UNKNOWN").upper()

                if confidence < 0.4:
                    continue  

                color = (0, 0, 255) if animal_name in DANGEROUS_ANIMALS else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{animal_name} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if animal_name in DANGEROUS_ANIMALS:
                    send_sms_alert(animal_name, x1, y1, width, height)
                    play_sound_alert()

    cv2.imshow("Wild Animal Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
