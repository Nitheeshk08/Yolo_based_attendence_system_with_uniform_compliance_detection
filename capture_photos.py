import cv2
import os

# Path to the database folder
database_path = "C:/Users/ASUS/Desktop/BATCH-8/Model_Training/Face_detection/database"

# Ask for student roll number
roll_no = input("Enter Student Roll Number: ").strip()

# Create folder for the student if it doesn't exist
student_folder = os.path.join(database_path, roll_no)
os.makedirs(student_folder, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

print(f"📸 Capturing images for Roll Number: {roll_no}... Press 'q' to stop.")

img_count = len(os.listdir(student_folder))  # Start from existing images

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture image")
        break

    # Show the webcam feed
    cv2.imshow("Press 's' to save, 'q' to quit", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Press 's' to save an image
        img_count += 1
        img_path = os.path.join(student_folder, f"{img_count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"✅ Image {img_count}.jpg saved!")

    elif key == ord('q'):  # Press 'q' to exit
        print("📤 Exiting...")
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
