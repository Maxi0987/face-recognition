import cv2  # Importiert die OpenCV-Bibliothek für Bild- und Videoverarbeitung
import face_recognition  # Importiert die face_recognition-Bibliothek für Gesichtserkennung
import os  # Importiert das os-Modul für Dateisystemoperationen

# Lade bekannte Gesichter und Labels
def load_face_encodings(folder_path, label):  # Definiert eine Funktion zum Laden von Gesichtseigenschaften aus einem Ordner
    encodings = []  # Erstellt eine leere Liste für die Gesichtseigenschaften
    for filename in os.listdir(folder_path):  # Iteriert über alle Dateien im angegebenen Ordner
        img_path = os.path.join(folder_path, filename)  # Erstellt den vollständigen Pfad zur Bilddatei
        image = face_recognition.load_image_file(img_path)  # Lädt das Bild als numpy-Array
        face_locations = face_recognition.face_locations(image)  # Findet die Positionen aller Gesichter im Bild
        if face_locations:  # Prüft, ob mindestens ein Gesicht gefunden wurde
            encoding = face_recognition.face_encodings(image, face_locations)[0]  # Berechnet die Gesichtseigenschaften für das erste gefundene Gesicht
            encodings.append((encoding, label))  # Fügt das Encoding und das zugehörige Label der Liste hinzu
    return encodings  # Gibt die Liste der Encodings und Labels zurück

good_faces = load_face_encodings('assets/Good', 'good')  # Lädt Encodings für "gute" Gesichter aus dem entsprechenden Ordner
bad_faces = load_face_encodings('assets/Bad', 'bad')  # Lädt Encodings für "schlechte" Gesichter aus dem entsprechenden Ordner
known_encodings = [e[0] for e in good_faces + bad_faces]  # Erstellt eine Liste aller bekannten Encodings
known_labels = [e[1] for e in good_faces + bad_faces]  # Erstellt eine Liste aller zugehörigen Labels

cap = cv2.VideoCapture(0)  # Öffnet die Webcam (Kamera mit Index 0)

while True:  # Startet eine Endlosschleife für die Videoverarbeitung
    ret, frame = cap.read()  # Liest ein Bild (Frame) von der Kamera
    if not ret:  # Prüft, ob das Bild erfolgreich gelesen wurde
        break  # Beendet die Schleife, falls kein Bild gelesen werden konnte

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Konvertiert das Bild von BGR (OpenCV-Standard) nach RGB (face_recognition erwartet RGB)
    face_locations = face_recognition.face_locations(rgb_frame)  # Findet die Positionen aller Gesichter im aktuellen Frame
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)  # Berechnet die Encodings für alle gefundenen Gesichter

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):  # Iteriert über alle gefundenen Gesichter und deren Encodings
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)  # Vergleicht das aktuelle Encoding mit allen bekannten Encodings
        label = "unknown face"  # Setzt das Standardlabel auf "unbekannt"
        if True in matches:  # Prüft, ob es eine Übereinstimmung gibt
            first_match_index = matches.index(True)  # Findet den Index der ersten Übereinstimmung
            label = known_labels[first_match_index]  # Holt das zugehörige Label
        color = (0, 255, 0) if label == "good" else (0, 0, 255) if label == "bad" else (0, 255, 255)  # Wählt die Farbe für das Rechteck je nach Label
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)  # Zeichnet ein Rechteck um das erkannte Gesicht
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Schreibt das Label über das Rechteck

    cv2.imshow('Gesichtserkennung', frame)  # Zeigt das aktuelle Bild mit den Markierungen im Fenster an

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Wartet auf Tastendruck; wenn 'q' gedrückt wird, wird die Schleife beendet
        break  # Beendet die Schleife

cap.release()  # Gibt die Kamera frei
cv2.destroyAllWindows()  # Schließt alle OpenCV-Fenster

# Diese Datei wird nicht mehr benötigt, da die Gesichtserkennung im Flask-Stream läuft.
# ...keine Änderungen notwendig, aber nicht mehr ausführen...