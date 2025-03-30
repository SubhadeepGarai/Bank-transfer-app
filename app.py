import os
import cv2
import numpy as np
import json
from flask import Flask, render_template, request, jsonify, session
from deepface import DeepFace

app = Flask(__name__)
app.secret_key = "your_secret_key"

db_file = "user_database.json"
if os.path.exists(db_file):
    with open(db_file, "r") as f:
        user_db = json.load(f)
else:
    user_db = {}

if os.path.exists("embeddings.npy"):
    embeddings = np.load("embeddings.npy", allow_pickle=True).item()
else:
    embeddings = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/verify_user', methods=['POST'])
def verify_user():
    data = request.json
    input_bank_name = data['bank_name']
    input_name = data['name']
    input_phone = data['phone']
    input_account = data['account_number']
    input_ifsc = data['ifsc_code']
    
    for user, details in user_db.items():
        if (
            details['bank_name'] == input_bank_name and
            details['name'] == input_name and
            details['phone'] == input_phone and
            details['account_number'] == input_account and
            details['ifsc_code'] == input_ifsc
        ):
            session['user'] = user
            return jsonify({"message": "Details matched. Proceed to PIN verification.", "status": "success"})
    
    return jsonify({"message": "Details do not match. Transaction denied.", "status": "fail"})

@app.route('/verify_pin', methods=['POST'])
def verify_pin():
    data = request.json
    user = session.get('user')
    if user and user_db[user]['pin'] == data['pin']:
        session['amount'] = data['amount']
        return jsonify({"message": "PIN verified. Proceed to face verification.", "status": "success"})
    
    return jsonify({"message": "Incorrect PIN. Transaction denied.", "status": "fail"})

@app.route('/recognize_faces', methods=['POST'])
def recognize_faces():
    user = session.get('user')
    if not user:
        return jsonify({"message": "User session expired.", "status": "fail"})
    
    file = request.files['image']
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    try:
        result = DeepFace.represent(image, model_name="Facenet", enforce_detection=False)
        if not result:
            return jsonify({"message": "Face not detected.", "status": "fail"})
        
        face_embedding = result[0]["embedding"]
        max_similarity = max(
            [np.dot(face_embedding, embed) / (np.linalg.norm(face_embedding) * np.linalg.norm(embed))
             for embed in embeddings.get(user, [])],
            default=-1
        )
        
        if max_similarity > 0.7:
            withdraw_amount = float(session.get('amount', 0))
            current_balance = float(user_db[user]['amount'])
            
            if withdraw_amount > current_balance:
                return jsonify({"message": "Insufficient balance!", "status": "fail"})
            
            user_db[user]['amount'] = str(current_balance - withdraw_amount)
            with open(db_file, "w") as f:
                json.dump(user_db, f, indent=4)
            
            return jsonify({"message": f"Withdrawal successful! New balance: {current_balance - withdraw_amount}", "status": "success"})
        else:
            return jsonify({"message": "Face not recognized. Transaction denied.", "status": "fail"})
    
    except Exception as e:
        return jsonify({"message": f"Error recognizing face: {e}", "status": "fail"})

if __name__ == '__main__':
    app.run(debug=True)
