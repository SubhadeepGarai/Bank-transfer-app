import os
import cv2
import numpy as np
import json
from deepface import DeepFace

# Load user database
db_file = "user_database.json"
if os.path.exists(db_file):
    with open(db_file, "r") as f:
        user_db = json.load(f)
else:
    print("User database not found.")
    user_db = {}

# Load face embeddings
if os.path.exists("embeddings.npy"):
    embeddings = np.load("embeddings.npy", allow_pickle=True).item()
else:
    print("No trained embeddings found. Train the dataset first.")
    embeddings = {}

def verify_user_details():
    print("Please enter your details:")
    input_bank_name = input("Enter your bank name: ")
    input_name = input("Enter your name: ")
    input_phone = input("Enter your phone number: ")
    input_account = input("Enter your account number: ")
    input_ifsc = input("Enter your IFSC code: ")
    
    for user, details in user_db.items():
        if (
            details['bank_name'] == input_bank_name and
            details['name'] == input_name and
            details['phone'] == input_phone and
            details['account_number'] == input_account and
            details['ifsc_code'] == input_ifsc
        ):
            print("Details matched. Proceeding to PIN verification...")
            return user, details
    
    print("Details do not match. Transaction denied.")
    return None, None

def recognize_faces(user):
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            try:
                # Get face embedding
                result = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)
                if not result:
                    continue
                
                face_embedding = result[0]["embedding"]
                max_similarity = -1
                
                # Verify face match
                for embed in embeddings.get(user, []):
                    similarity = np.dot(face_embedding, embed) / (np.linalg.norm(face_embedding) * np.linalg.norm(embed))
                    if similarity > max_similarity:
                        max_similarity = similarity
                
                if max_similarity > 0.7:
                    print("Face verified successfully. Transaction approved.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("Face not recognized. Transaction denied.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
                
            except Exception as e:
                print(f"Error recognizing face: {e}")
        
        cv2.imshow("Recognize Faces", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return False

if __name__ == "__main__":
    user, user_data = verify_user_details()
    if user:
        withdraw_amount = float(input("Enter amount to withdraw: "))
        current_balance = float(user_data["amount"])
        
        if withdraw_amount > current_balance:
            print("Insufficient balance!")
        else:
            pin = input("Enter your 4-digit PIN: ")
            if pin == user_data["pin"]:
                print("PIN verified. Proceeding to face verification...")
                if recognize_faces(user):
                    new_balance = current_balance - withdraw_amount
                    user_db[user]["amount"] = str(new_balance)
                    
                    # Update the database
                    with open(db_file, "w") as f:
                        json.dump(user_db, f, indent=4)
                    
                    print(f"Withdrawal successful! New balance: {new_balance}")
            else:
                print("Incorrect PIN. Transaction denied.")
