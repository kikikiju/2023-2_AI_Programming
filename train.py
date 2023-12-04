import os
import cv2
import dlib
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# 환경 변수 설정
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 데이터 경로 설정
base_folder = 'user_data'
train_images = []
train_labels = []
val_images = []
val_labels = []

# 모든 사용자 폴더에 대해
for user_folder in os.listdir(base_folder):
    if user_folder.startswith('user_'):
        user_id = int(user_folder.split('_')[1])

        # 훈련 데이터 불러오기 및 전처리
        train_data_path = os.path.join(base_folder, user_folder, 'train_data')

        for filename in os.listdir(train_data_path):
            if filename.endswith('.jpg'):
                img = cv2.imread(os.path.join(train_data_path, filename), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (180, 180))
                train_images.append(img)
                train_labels.append(user_id)
        
        # 검증 데이터 불러오기 및 전처리
        val_data_path = os.path.join(base_folder, user_folder, 'val_data')

        for filename in os.listdir(val_data_path):
            if filename.endswith('.jpg'):
                img = cv2.imread(os.path.join(val_data_path, filename), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (180, 180))
                val_images.append(img)
                val_labels.append(user_id)

train_scaled = np.array(train_images).reshape(-1, 180, 180, 1) / 255.0
train_labels = np.array(train_labels)

val_scaled = np.array(val_images).reshape(-1, 180, 180, 1) / 255.0
val_labels = np.array(val_labels)

# 모델 정의
model = keras.Sequential ()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(len(set(train_labels)), activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam', loss=tf.compat.v1.losses.sparse_softmax_cross_entropy, metrics=['accuracy'])

# 모델 학습
model.fit(train_scaled, train_labels, epochs=20, validation_data=(val_scaled, val_labels))

# 모델 저장
model.save('cnn-model.keras')