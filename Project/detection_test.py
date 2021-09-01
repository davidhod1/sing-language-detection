from detection_functions import *
import keras

sequence_prediction = []
sentence = []
predictions = []
threshold = 0.5

model = keras.models.load_model('sign_lang_detection.h5')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

with mp_holistic.Holistic(min_detection_confidence = 0.7, min_tracking_confidence = 0.7) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()

        image, results = mp_integration(frame, holistic)

        keypoints = extract_points(results)
        sequence_prediction.append(keypoints)
        sequence_prediction = sequence_prediction[-60:]

        if len(sequence_prediction) == 60:
            res = model.predict(np.expand_dims(sequence_prediction, axis=0))[0]
            print(res)

            if (res[np.argmax(res)] > 0.7):
                prob = res[np.argmax(res)]
                cv2.putText(image, '{} '.format(sign_actions[np.argmax(res)]), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, '{} '.format(prob), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                print(sign_actions[np.argmax(res)])
                print(res)
            else:
                print("NOne")

        cv2.imshow('OpenCv Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()