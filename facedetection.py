import cv2
from simple_facerec import SimpleFacerec

sfr = SimpleFacerec()
sfr.load_encoding_images("images")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_locations, face_names = sfr.detect_known_faces(frame)

    for (y1, x2, y2, x1), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 200), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
