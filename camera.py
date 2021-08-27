import cv2


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    image = cv2.flip(frame, 1)

    cv2.imshow("Camera", image)

    if cv2.waitKey(1) == ord('1'):
        break

cap.release()
cv2.destroyAllWindows()