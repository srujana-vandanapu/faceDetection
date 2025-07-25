import face_recognition
import cv2
import os
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def load_encoding_images(self, images_path):
        images_path = os.path.abspath(images_path)
        files = os.listdir(images_path)

        for file_name in files:
            img_path = os.path.join(images_path, file_name)
            img = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(img)
            if len(encoding) > 0:
                self.known_face_encodings.append(encoding[0])
                self.known_face_names.append(os.path.splitext(file_name)[0])

    def detect_known_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []

        for encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, encoding)
            name = "Unknown"

            if len(matches) > 0:
                face_distances = face_recognition.face_distance(self.known_face_encodings, encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        return face_locations, face_names
