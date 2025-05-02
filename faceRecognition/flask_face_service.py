from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2
import base64
import os

app = Flask(__name__)

# 存储已知用户信息的目录
KNOWN_FACES_DIR = "known_faces"

def load_known_faces(user_id):
    if user_id:
        path = KNOWN_FACES_DIR + f"/{user_id}"
    else:
        path = KNOWN_FACES_DIR + f"/admin"
    known_faces = {}
    if not os.path.exists(path):
        os.makedirs(path)
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            name = os.path.splitext(filename)[0]
            if user_id:
                if name == user_id:
                    image_path = os.path.join(path, filename)
                    image = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(image)[0]
                    known_faces[user_id] = encoding
            else:
                image_path = os.path.join(path, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_faces[name] = encodings[0]
    return known_faces

def verify_face(image_base64, known_faces):
    # 解码Base64图像
    image_data = base64.b64decode(image_base64.split(",")[1])
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 提取人脸特征
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        return {"status": "1", "message": "No face detected"}

    face_encoding = face_recognition.face_encodings(image, face_locations)[0]

    # 与已知用户比对
    for user_id, known_encoding in known_faces.items():
        match = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.5)
        if match[0]:
            return {"status": "0", "message": f"Verification successful: The user is {user_id}", "userid": user_id}

    return {"status": "1", "message": "Verification failed: The user is not recognized"}

def save_face(image_base64, user_id):
    # 解码Base64图像
    image_data = base64.b64decode(image_base64.split(",")[1])
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 保存图像到已知用户目录
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
    image_path = os.path.join(KNOWN_FACES_DIR + f"/{user_id}", f"{user_id}.jpg")
    cv2.imwrite(image_path, image)

    return {"status": "success", "message": f"Face saved for user {user_id}"}

@app.route("/register-face", methods=["POST"])
def register_face():
    data = request.json
    image_base64 = data.get("image")
    user_id = data.get("userId")
    if not image_base64 or not user_id:
        return jsonify({"status": "1", "message": "Missing image or userId"}), 400
    result = save_face(image_base64, user_id)
    return jsonify(result)

@app.route("/verify-face", methods=["POST"])
def verify_face_endpoint():
    data = request.json
    image_base64 = data.get("image")
    user_id = data.get("userId")
    if not image_base64:
        return jsonify({"status": "1", "message": "Missing image"}), 400
    known_faces = load_known_faces(user_id)
    result = verify_face(image_base64, known_faces)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)