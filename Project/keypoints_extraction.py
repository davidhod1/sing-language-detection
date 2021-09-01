from detection_functions import *

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
                    cv2.putText(image, 'Collecting frames for sign language action: {}  Frame number: {}'.format(action, sequence), (30, 50),
                                cv2.FONT_HERSHEY_PLAIN,2, (0,0, 255), 2, cv2.LINE_AA)
                    cv2.waitKey(2000)
                elif sequence == num_of_sequences - 1:
                    cv2.putText(image, 'Last sequence of sign action: {}'.format(action),(30, 50),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_4)
                else:
                    cv2.putText(image, 'Collecting frames for sign language action: {}  Frame number: {}'.format(action, sequence), (30, 50),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0,0, 255), 2, cv2.LINE_4)

                keypoints = extract_points(result)
                np_file = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(np_file, keypoints)

                if sequence == num_of_sequences - 1 and frame_num == sequence_length - 1:
                    cv2.waitKey(5000)

                cv2.imshow("Camera Feed", image)

                key = cv2.waitKey(10)
                if key == ord('q'):
                    break
                if key == ord('p'):
                    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()