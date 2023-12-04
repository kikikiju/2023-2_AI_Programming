import os
import cv2
import dlib
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

model = load_model("cnn-model.keras")

# 얼굴 감지기 초기화
detector = dlib.get_frontal_face_detector()

# 웹캠에서 실시간 처리
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    # 이미지를 흑백으로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = gray[y:y+h, x:x+w]
        
        # 얼굴 크기를 모델에 맞게 조정
        face_img = cv2.resize(face_img, (180, 180))
        face_img = np.expand_dims(face_img, axis=-1)
        face_img = np.expand_dims(face_img, axis=0)

        # 얼굴 인식 모델 적용
        prediction = model.predict(face_img)

        # 결과 출력
        user_id = np.argmax(prediction)
        result = f"User: {user_id + 1}"
        cv2.putText(frame, result, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # 얼굴 주변에 사각형 그리기
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 화면에 결과 출력
    cv2.imshow('Real-time Face Recognition', frame)

    # 'Esc' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 작업이 완료되면 해제
cap.release()
cv2.destroyAllWindows()