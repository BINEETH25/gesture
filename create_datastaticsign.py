import os
import cv2
import mediapipe as mp
import numpy as np
import pickle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
DATA_DIR = './data_staticsign'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = np.zeros(126)  # Initialize with zeros for two hands (21*3*2)

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if i < 2:  # Ensure only two hands are processed
                    for j, landmark in enumerate(hand_landmarks.landmark):
                        # Fill data_aux with landmark coordinates
                        idx = i * 63 + j * 3  # Calculate index in the array
                        data_aux[idx] = landmark.x
                        data_aux[idx + 1] = landmark.y
                        data_aux[idx + 2] = landmark.z

        data.append(data_aux)
        labels.append(dir_)

# Save the data
with open('data_staticsign.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
