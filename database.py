import os
import cv2
import numpy as np
import json
from deepface import DeepFace

# Directory to store the dataset
data_dir = "face_dataset"
os.makedirs(data_dir, exist_ok=True)

# File to store user details
db_file = "user_database.json"
if not os.path.exists(db_file):
    with open(db_file, "w") as f:
        json.dump({}, f)

def create_face_dataset():
    bank_name = input("Enter bank name: ")
    name = input("Enter full name: ")
    phone = input("Enter phone number: ")
    account_number = input("Enter account number: ")
    ifsc_code = input("Enter IFSC code: ")
    amount = input("Enter account balance: ")
    pin = input("Enter a 4-digit PIN: ")
    
    if not pin.isdigit() or len(pin) != 4:
        print("Invalid PIN. It must be a 4-digit number.")
        return
    
    person_id = f"{name}_{account_number}"
    person_dir = os.path.join(data_dir, person_id)
    os.makedirs(person_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    print("Capturing images. Press 'q' to quit.")
    count = 0
    
    while count < 20:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            count += 1
            face_img = frame[y:y + h, x:x + w]
            face_path = os.path.join(person_dir, f"{person_id}_{count}.jpg")
            cv2.imwrite(face_path, face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("Capture Faces", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} images for {name}.")
    
    # Save user details to JSON database
    with open(db_file, "r") as f:
        user_db = json.load(f)
    
    user_db[person_id] = {
        "bank_name": bank_name,
        "name": name,
        "phone": phone,
        "account_number": account_number,
        "ifsc_code": ifsc_code,
        "amount": amount,
        "pin": pin
    }
    
    with open(db_file, "w") as f:
        json.dump(user_db, f, indent=4)
    
    print("User details saved successfully.")

def train_face_dataset():
    embeddings = {}
    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        if os.path.isdir(person_dir):
            embeddings[person] = []
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                try:
                    result = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)
                    if result:
                        embedding = result[0]["embedding"]
                        embeddings[person].append(embedding)
                except Exception as e:
                    print(f"Failed to process {img_path}: {e}")
    np.save("embeddings.npy", embeddings)
    print("Embeddings saved.")

if __name__ == "__main__":
    print("1. Create Face Dataset\n2. Train Face Dataset")
    choice = input("Enter your choice: ")
    
    if choice == "1":
        create_face_dataset()
    elif choice == "2":
        train_face_dataset()
    else:
        print("Invalid choice.")
