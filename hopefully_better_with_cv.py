import cv2
import face_recognition
import os

# Lade bekannte Gesichter und Labels
def load_face_encodings(folder_path, label):
    encodings = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        if face_locations:
            encoding = face_recognition.face_encodings(image, face_locations)[0]
            encodings.append((encoding, label))
    return encodings

good_faces = load_face_encodings('assets/Good', 'good')
bad_faces = load_face_encodings('assets/Bad', 'bad')
known_encodings = [e[0] for e in good_faces + bad_faces]
known_labels = [e[1] for e in good_faces + bad_faces]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        label = "unknown face"
        if True in matches:
            first_match_index = matches.index(True)
            label = known_labels[first_match_index]
        color = (0, 255, 0) if label == "good" else (0, 0, 255) if label == "bad" else (0, 255, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Gesichtserkennung', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()