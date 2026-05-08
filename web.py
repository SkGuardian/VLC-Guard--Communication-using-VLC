import cv2
import serial
import time
from ultralytics import YOLO

# --- 1. CONFIGURATION & MATH SETUP ---
# We store the real-world widths (in cm) of different objects so the math is accurate
OBJECT_WIDTHS = {
    'car': 19.5,
    'truck': 19.5,
    'cell phone': 7.0,  # Perfect for testing at your desk!
}
FOCAL_LENGTH = 600 # Adjust this to calibrate your exact webcam

# --- 2. ARDUINO SETUP (Commented out for testing) ---
# arduino_port = 'COM3' 
# arduino = serial.Serial(arduino_port, 9600, timeout=1)
# time.sleep(2)

# --- 3. LOAD THE YOLOv8 AI MODEL ---
print("Downloading and loading AI model... (This takes a few seconds the first time)")
# 'yolov8n.pt' is the "Nano" version. Extremely fast and lightweight.
model = YOLO('yolov8n.pt') 

cap = cv2.VideoCapture(0)
print("Starting AI Radar. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run the YOLO detection on the frame
    # verbose=False stops it from spamming your console with text
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            # Get the name of the object detected
            class_id = int(box.cls[0])
            object_name = model.names[class_id]

            # Only process objects that are in our OBJECT_WIDTHS list
            if object_name in OBJECT_WIDTHS:
                
                # Get the pixel coordinates of the bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1 # Calculate pixel width
                
                # Calculate actual distance using the specific object's real-world width
                known_width = OBJECT_WIDTHS[object_name]
                distance_cm = (known_width * FOCAL_LENGTH) / w

                # Draw the box and label
                cv2.putText(frame, f"{object_name.upper()} | {round(distance_cm, 1)} cm", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # --- 4. THE 3-STAGE TRIGGER LOGIC ---
                if distance_cm < 30:
                    # EMERGENCY
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) 
                    cv2.putText(frame, "EMERGENCY: BRAKE!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    # arduino.write(b'2') 
                    
                elif 30 <= distance_cm <= 60:
                    # WARNING
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2) 
                    cv2.putText(frame, "WARNING: SLOW DOWN", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    # arduino.write(b'1') 
                    
                else:
                    # SAFE
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                    cv2.putText(frame, "SAFE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # arduino.write(b'0') 

    cv2.imshow("ADAS YOLOv8 Radar", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# arduino.close()