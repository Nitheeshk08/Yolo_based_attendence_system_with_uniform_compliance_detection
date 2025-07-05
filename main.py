import cv2
import os
import csv
import datetime
import numpy as np
import face_recognition  # Faster than DeepFace
import pickle
from ultralytics import YOLO
from dotenv import load_dotenv
from twilio.rest import Client
import pandas as pd

# Load API Key from .env file
load_dotenv()
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Initialize Twilio Client
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# ---------------- Step 1: Load Face Encodings ---------------- #
def load_encodings():
    if os.path.exists("face_encodings.pkl"):
        with open("face_encodings.pkl", "rb") as f:
            return pickle.load(f)
    return {}

# ---------------- Step 2: Fast Face Recognition ---------------- #
def recognize_face(face_roi, known_encodings, threshold=0.5):
    face_encodings = face_recognition.face_encodings(face_roi)
    if not face_encodings:
        return "Unknown"

    face_encoding = face_encodings[0]
    distances = {id_: face_recognition.compare_faces(face_encoding - enc) for id_, enc in known_encodings.items()}
    best_match = min(distances, key=distances.get)
    
    return best_match if distances[best_match] < threshold else "Unknown"


# ---------------- Step 3: Attendance System ---------------- #
db_path = r"DATABASE ABSOLUTE PATH"     #Give databse absolute path here
encodings = load_encodings()

yolo_model = YOLO("best.pt")   #put your custom model path here.


#You may have a CSV file of all the class members using which we are going to create attendece file
#Load studetn details
roll_number_to_details = {}
with open("Class.csv", mode="r") as file:
    reader = csv.reader(file)
    headers = next(reader)  # Read column headers
    for row in reader:
        row_dict = dict(zip(headers, row))  # Convert to dictionary
        roll_number_to_details[row[0]] = (row[1], row_dict["Phone No"])  # (Name, Phone Num

cap = cv2.VideoCapture(0)
recorded_students = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Detect uniform and ID card using YOLO
    id_card_detected = False
    uniform_bboxes = []
    id_card_bbox = None
    results = yolo_model(frame)

    if results:
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = "Uniform" if box.cls[0] == 0 else "No Uniform"
                
                if box.cls[0] == 1:  # ID card class
                    id_card_bbox = (x1, y1, x2, y2)
                else:
                    uniform_bboxes.append((x1, y1, x2 - x1, y2 - y1, label))

    if id_card_bbox:
        for (ux, uy, uw, uh, _) in uniform_bboxes:
            if (id_card_bbox[0] >= ux and id_card_bbox[1] >= uy and
                id_card_bbox[2] <= (ux + uw) and id_card_bbox[3] <= (uy + uh)):
                id_card_detected = True
                break

    attendance_data = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        roll_no = "Unknown"
        distances = {id_: np.linalg.norm(face_encoding - enc) for id_, enc in encodings.items()}
        best_match = min(distances, key=distances.get)
        if distances[best_match] < 0.5:
            roll_no = best_match

        if roll_no != "Unknown" and roll_no not in recorded_students:
            name, mobile = roll_number_to_details.get(roll_no, ("Unknown", "Unknown"))
            date = datetime.datetime.now().strftime("%Y-%m-%d")
            time = datetime.datetime.now().strftime("%H:%M:%S")
            uniform_status = "Uniform" if id_card_detected else "No Uniform"
            id_card_status = "Yes" if id_card_detected else "No"

            attendance_data.append([roll_no, name, uniform_status, id_card_status, mobile, date, time])
            recorded_students.add(roll_no)

            # Send SMS if student is out of uniform
            if uniform_status == "No Uniform" or id_card_status == "No":
                # payment_link = f"https://yourwebsite.com/payfine?roll_no={roll_no}"  if you have a payment gateway you can implement this part too
                message_body = (f"Hello {name}, you did not wear the proper uniform today. "
                                f"A fine has been applied. Pay here: {payment_link}")
                try:
                    client.messages.create(
                        body=message_body,
                        from_=TWILIO_PHONE_NUMBER,
                        to=f"+91{mobile}"
                    )
                    print(f"✅ SMS sent to {name} ({mobile})")
                except Exception as e:
                    print(f"❌ Failed to send SMS to {name} ({mobile}) - {e}")

    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    csv_filename = f"attendance_{today_date}.csv"

    if attendance_data:
        with open(csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["Roll No", "Name", "Uniform Status", "ID Card", "Phone No", "Date", "Time"])
            writer.writerows(attendance_data)
        print(f"✅ Attendance recorded in {csv_filename}: {attendance_data}")
    
    # Display bounding boxes
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, roll_no, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for (x, y, w, h, label) in uniform_bboxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(frame, f"ID Card: {'Detected' if id_card_detected else 'Not Detected'}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
