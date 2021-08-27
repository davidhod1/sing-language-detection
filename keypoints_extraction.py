import cv2
import numpy as np
import os
import mediapipe as mp
import keyboard
import time

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def mp_integration(image, model):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    draw_landmarks(image, results)
    return image, results

def extract_points(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(132)

    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(1404)

    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(63)

    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(63)

    return np.concatenate([pose, face, left_hand, right_hand])



mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


DATA_PATH = os.path.join("Dataset")

sign_actions = np.array(['Pozdrav'])
num_of_sequences = 35
sequence_length = 35

for action in sign_actions:
    for sequence in range(num_of_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass



cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in sign_actions:
        for sequence in range(num_of_sequences):
            for frame_num in range(sequence_length):

                ret, frame = cap.read()

                image, result = mp_integration(frame, holistic)

                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (100,150), cv2.FONT_HERSHEY_PLAIN, 2, (0,0, 255), 4, cv2.LINE_4)
                    cv2.putText(image, 'Collecting frames for sign language action: {}  Frame number: {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_PLAIN,1, (0,0, 255), 1, cv2.LINE_AA)
                    cv2.waitKey(2000)
                elif sequence == num_of_sequences - 1:
                    cv2.putText(image, 'Last sequence of sign action: {}'.format(action),(15, 12),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_4)
                else:
                    cv2.putText(image, 'Collecting frames for sign language action: {}  Frame number: {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0,0, 255), 1, cv2.LINE_4)

                keypoints = extract_points(result)
                np_file = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(np_file, keypoints)

                cv2.imshow("Camera Feed", image)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                if key == ord('p'):
                    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()