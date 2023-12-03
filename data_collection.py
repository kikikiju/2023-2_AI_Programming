import cv2
import dlib
import os

# 얼굴 감지기 초기화
detector = dlib.get_frontal_face_detector()

# 웹캠에서 실시간 처리
cap = cv2.VideoCapture(0)

# 이미지 저장 경로
base_folder = 'user_data'
user_id = 0

while os.path.exists(os.path.join(base_folder, f'user_{user_id}')):
    user_id += 1

user_data_path = os.path.join(base_folder, f'user_{user_id}')
train_data_path = os.path.join(user_data_path, 'train_data')
val_data_path = os.path.join(user_data_path, 'val_data')

# 폴더 생성
os.makedirs(train_data_path)
os.makedirs(val_data_path)

# 이미지 저장을 위한 카운터
train_img_count = 0
val_img_count = 0

while True:
    ret, frame = cap.read()

    # 얼굴 감지
    faces = detector(frame)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = frame[y:y+h, x:x+w]

        # 얼굴 이미지를 저장
        if train_img_count < 80:
            img_filename = os.path.join(train_data_path, f'face_{train_img_count + 1}.jpg')
            train_img_count += 1
        elif val_img_count < 20:
            img_filename = os.path.join(val_data_path, f'face_{val_img_count + 1}.jpg')
            val_img_count += 1
        else:
            cap.release()
            cv2.destroyAllWindows()
            exit(0)  # 80장 + 20장 저장 후 종료

        # 이미지를 흑백으로 변환하여 저장
        cv2.imwrite(img_filename, cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY))
        print(f"Saved: {img_filename}")

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