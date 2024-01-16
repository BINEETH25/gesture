import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model_staticsign.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

labels_dict = {0: 'Airplane', 1: 'Ship', 2: 'Car',3 : 'Taxi',4 : 'Bus',5 : 'Cover',
               6: 'Key', 7: 'Lock', 8: 'Telephone',9: 'Father', 10: 'Mother', 11: 'Family',
               12: 'Water', 13: 'Time', 14: 'Home', 15: 'Love', 16: 'Money', 17: 'Pray',
               18: 'Church', 19: 'I Love You', 20: 'I Hate You', 21: 'No', 22: 'Yes', 23: 'I am',
               24: 'Fine', 25:'Okay', 26: 'Sorry', 27: 'Hello', 28: 'Help', 29: 'Hungry',
               30: 'Stand', 31: 'Calm down', 32: 'Stop', 33: 'Where', 34: 'Why'}

last_predicted_character = None  # Variable to store the last predicted gesture

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = np.zeros(126)  # 63 features per hand, 2 hands
    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            x_, y_ = [], []
            if i < 2:  # Process max 2 hands
                for j, landmark in enumerate(hand_landmarks.landmark):
                    idx = i * 63 + j * 3  # Calculate index in data_aux
                    data_aux[idx] = landmark.x
                    data_aux[idx + 1] = landmark.y
                    data_aux[idx + 2] = landmark.z

                    # Collecting x and y for bounding box
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Draw bounding box
                x1, y1 = int(min(x_) * W), int(min(y_) * H)
                x2, y2 = int(max(x_) * W), int(max(y_) * H)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

    # Make a prediction and display it only if a hand is detected
    if hand_detected:
        prediction = model.predict([data_aux])
        predicted_character = labels_dict[int(prediction[0])]

        # Check if the predicted gesture is different from the last one
        if predicted_character != last_predicted_character:
            print(predicted_character)
            last_predicted_character = predicted_character

        # Display the prediction
        cv2.putText(frame, predicted_character, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()