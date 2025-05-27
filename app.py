from flask import Flask, render_template, Response, redirect, url_for, session, request, jsonify
import cv2
import face_recognition
import os
import base64
from datetime import datetime
import numpy as np  # Muss ganz oben stehen!

app = Flask(__name__, template_folder='templates')
app.secret_key = 'dein_geheimer_schluessel'  # Für Session-Management

login_status = {}

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

# Video-Stream-Generator
def gen_frames(client_id):
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
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
                if label == "good":
                    login_status[client_id] = True
                    # session['logged_in'] = True  # Setze Session erst hier!
            color = (0, 255, 0) if label == "good" else (0, 0, 255) if label == "bad" else (0, 255, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Wenn eingeloggt, Stream beenden
        if login_status.get(client_id):
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    # Entferne die automatische Weiterleitung auf dashboard
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    client_id = request.remote_addr
    return Response(gen_frames(client_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('index'))
    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    # Setze auch den Login-Status für den Client zurück
    client_id = request.remote_addr
    login_status.pop(client_id, None)
    return redirect(url_for('index'))

@app.route('/status')
def status():
    client_id = request.remote_addr
    # Setze session['logged_in'] erst hier, wenn login_status gesetzt ist
    if login_status.get(client_id):
        session['logged_in'] = True
        login_status.pop(client_id, None)
        return jsonify({'logged_in': True})
    return jsonify({'logged_in': bool(session.get('logged_in'))})

@app.route('/register_face', methods=['POST'])
def register_face():
    print("Register Face: Request received")  # Debug
    data = request.get_json()
    if not data or 'image' not in data:
        print("Register Face: No image in request")  # Debug
        return jsonify({'success': False, 'message': 'Kein Bild empfangen.'})

    try:
        # Bilddaten dekodieren
        print("Register Face: Decoding image data")  # Debug
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            print("Register Face: Frame is None after decoding")  # Debug
            return jsonify({'success': False, 'message': 'Bild konnte nicht verarbeitet werden.'})

        print("Register Face: Running face detection")  # Debug
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        print(f"Register Face: Detected faces: {len(face_locations)}")  # Debug

        if len(face_locations) == 0:
            print("Register Face: No face detected")  # Debug
            return jsonify({'success': False, 'message': 'Kein Gesicht erkannt!'})
        if len(face_locations) > 1:
            print("Register Face: More than one face detected")  # Debug
            return jsonify({'success': False, 'message': 'Bitte nur ein Gesicht im Bild!'})

        # Gesicht speichern (Verzeichnis anlegen, falls nicht vorhanden)
        save_dir = os.path.join('assets', 'Good')
        if not os.path.exists(save_dir):
            print(f"Register Face: Creating directory {save_dir}")  # Debug
            os.makedirs(save_dir)
        filename = datetime.now().strftime("face_%Y%m%d_%H%M%S.jpg")
        save_path = os.path.join(save_dir, filename)
        print(f"Register Face: Saving image to {save_path}")  # Debug
        success = cv2.imwrite(save_path, frame)
        if not success:
            print("Register Face: Error saving image")  # Debug
            return jsonify({'success': False, 'message': 'Fehler beim Speichern des Bildes.'})
        print("Register Face: Success!")  # Debug
        return jsonify({'success': True, 'message': 'Gesicht erfolgreich registriert!'})
    except Exception as e:
        print(f"Register Face: Exception occurred: {e}")  # Debug
        return jsonify({'success': False, 'message': f'Fehler: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
