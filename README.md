# 🦊 Wildlife Intrusion Detection System  

AI-powered system for real-time wildlife intrusion detection using **YOLO** and **OpenCV**.  
Detects animals, filters false positives, and triggers alerts via sound or SMS to prevent human-wildlife conflict.  

---

## 📌 Features  
- **Real-Time Detection** – Detects animals instantly from live camera feed or video.  
- **False Positive Filtering** – Confidence and size thresholding + frame history tracking.  
- **Alerts & Notifications** – Plays alarm sound and optionally sends SMS using Twilio.  
- **Lightweight & Portable** – Works with webcams, CCTV, or IP cameras; can be deployed on PC or Raspberry Pi.  

---

## 🛠 Tech Stack  
- **Language:** Python  
- **Libraries:** OpenCV, YOLO (Ultralytics), Pygame, Twilio API  
- **Hardware:** Webcam, CCTV/IP Camera, or any video source  

---

## 📂 Project Structure  
```bash
├── main.py              # Main detection script
├── requirements.txt     # Python dependencies
├── alert.wav            # Alarm sound
└── README.md            # Project documentation
