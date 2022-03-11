from flask import Flask, request, send_from_directory
import face_recognition
import os
from PIL import Image

from keras.models import load_model
import cv2
import numpy as np
import sys

from datetime import datetime
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True,max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

app = Flask(__name__, static_url_path='')
#Load the saved model file
model = load_model("gesture-model05_20.h5")

uploads_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'uploads')
try:
    os.makedirs(uploads_dir)
except:
    pass

@app.route("/face_recognition",methods=["POST"])
def face_recon():
    files = list(request.files.items())
    if len(files) != 2:
        return {"status": "error", "message": "Please upload two images"}
    
    for i in range(len(files)):
        f = request.files[files[i][0]]
        f.save(os.path.join(uploads_dir, f.filename))
    
    original_pic = face_recognition.load_image_file(f"./uploads/{files[0][1].filename}")
    org_encode = face_recognition.face_encodings(original_pic)[0]

    try:
        stream_pic = face_recognition.load_image_file(f"./uploads/{files[1][1].filename}")
        stream_location = face_recognition.face_locations(stream_pic)
        stream_encode = face_recognition.face_encodings(stream_pic, stream_location)[0]
        # stream_encode = face_recognition.face_encodings(stream_pic)[0]

        results = face_recognition.compare_faces([org_encode], stream_encode)
        
        for i in range(len(files)):
            profile = request.files[files[i][0]]
            os.remove(os.path.join(uploads_dir, profile.filename))
        
        return {"status":"success" if results[0] else "error", "message": f"{results[0]}"}
    except Exception as e:
        if isinstance(e,IndexError):
            return {"status":"error","message":"Face not Found in webcam"}
        return {"status":"error","message":f"{e}"}

@app.route("/predict",methods=["POST"])
def predict():
    files = list(request.files.items())
    if len(files) != 1:
        return {"status": "error", "message": "Please upload one image"}

    f = files[0][1]
    filename = f.filename
    filename = filename.split(".")[0] + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + "." + filename.split(".")[1] 
    f.save(os.path.join(uploads_dir, filename))
    
    CATEGORY_MAP = {
        "blank": 0,
        "fist": 1,
        "five": 2,
        "ok": 3,
        "thumbsdown": 4,
        "thumbsup": 5
    }

    # This function returns the gesture name from its numeric equivalent 
    def mapper(val):
        return list(CATEGORY_MAP.keys())[list(CATEGORY_MAP.values()).index(val)]

    def calc_bounding_rect(image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)
        
        x, y, w, h = cv2.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    def get_hand(file):

        img = cv2.imread(file)
        frame = cv2.flip(img, 1)
        x , y, c = frame.shape
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(framergb)
        try:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks,result.multi_handedness):
                x1,y1,x2,y2 = calc_bounding_rect(framergb,hand_landmarks)
                cropped_image = frame[y1-50:y2+50,x1-50:x2+50]
                cv2.imwrite(file, cropped_image)
            return True
        except Exception as e:
            pass

        return False
    
    hand_detect = get_hand(f"./uploads/{filename}")
    print(hand_detect)
    if not hand_detect:
        return {"status":"error","message":"Hand not detected"}

    # Ensuring the input image has same dimensions that is used during training. 
    img = cv2.imread(f"./uploads/{filename}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (225, 225))

    # Predict the gesture from the input image
    prediction = model.predict(np.array([img]))
    gesture_numeric = np.argmax(prediction[0])
    gesture_name = mapper(gesture_numeric)
    print(gesture_name)
    
    # remove the saved file
    os.remove(os.path.join(uploads_dir, filename))
    
    return {"status":"success", "message": gesture_name}

@app.route('/ml/<path:path>')
def send_file(path):
    return send_from_directory('.', path)
if   __name__ == "__main__" :
    app.run(host="0.0.0.0",port=5000,debug=True)

