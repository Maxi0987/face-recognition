import os
import base64
from datetime import datetime
import numpy as np
import mysql.connector
import random
import logging
import threading
from flask import Flask, render_template, Response, redirect, url_for, session, request, jsonify, make_response
import cv2
import face_recognition
from decimal import Decimal

app = Flask(
    __name__,
    template_folder='templates',
    static_folder='static'
)
app.secret_key = 'dein_geheimer_schluessel'

login_status = {}
face_user_map = {}  # encoding.tobytes() -> user_id

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'gambling'
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def load_face_encodings(folder_path):
    encodings = []
    for filename in os.listdir(folder_path):
        if not (filename.endswith('.jpg') or filename.endswith('.png')):
            continue
        img_path = os.path.join(folder_path, filename)
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        if face_locations:
            encoding = face_recognition.face_encodings(image, face_locations)[0]
            # Extrahiere user_id aus Dateinamen: z.B. 1.png oder 1.jpg
            name_part = os.path.splitext(filename)[0]
            if name_part.isdigit():
                user_id = int(name_part)
                encodings.append((encoding, user_id))
                face_user_map[encoding.tobytes()] = user_id
            else:
                # Fallback für alte Namenskonvention: face_{user_id}_timestamp.jpg
                parts = filename.split('_')
                if len(parts) >= 3 and parts[1].isdigit():
                    user_id = int(parts[1])
                    encodings.append((encoding, user_id))
                    face_user_map[encoding.tobytes()] = user_id
    return encodings

def reload_face_encodings():
    global good_faces, known_encodings, known_user_ids
    good_faces = load_face_encodings('assets/Good')
    known_encodings = [e[0] for e in good_faces]
    known_user_ids = [e[1] for e in good_faces]

reload_face_encodings()

# Verwende eine globale Kamera, um Probleme beim erneuten 
# Öffnen zu vermeiden
camera_lock = threading.Lock()
camera = None

def get_camera():
    """Liefert ein einzelnes Camera-Objekt."""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
        return camera

# Video-Stream-Generator
def gen_frames(client_id):
    cap = get_camera()
    if not cap.isOpened():
        # Return a placeholder image if camera is not available
        error_img = np.full((240, 320, 3), 200, dtype=np.uint8)
        cv2.putText(error_img, "No Camera", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        _, buffer = cv2.imencode('.jpg', error_img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return
    try:
        while True:
            with camera_lock:
                success, frame = cap.read()
            if not success:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                label = "unknown face"
                user_id = None
                if True in matches:
                    first_match_index = matches.index(True)
                    user_id = known_user_ids[first_match_index]
                    label = get_username_by_id(user_id)
                    if user_id:
                        login_status[client_id] = user_id
                color = (0, 255, 0) if user_id else (0, 255, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Wenn eingeloggt, Stream beenden
            if login_status.get(client_id):
                break

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        # Kamera nicht sofort freigeben, um erneute
        # Initialisierungen zu vermeiden
        pass

@app.route('/video_feed')
def video_feed():
    client_id = request.remote_addr
    response = Response(gen_frames(client_id), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response

def get_username_by_id(user_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result:
            return result[0]
    except Exception as e:
        print("DB error:", e)
    return "unknown"

# Logging (wie im Gambling-Projekt)
os.makedirs('logs', exist_ok=True)
auth_logger = logging.getLogger('auth')
auth_logger.setLevel(logging.INFO)
auth_handler = logging.FileHandler(os.path.join('logs', 'auth.log'))
auth_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
auth_logger.addHandler(auth_handler)
game_logger = logging.getLogger('game')
game_logger.setLevel(logging.INFO)
game_handler = logging.FileHandler(os.path.join('logs', 'game.log'))
game_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
game_logger.addHandler(game_handler)
error_logger = logging.getLogger('error')
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler(os.path.join('logs', 'errors.log'))
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
error_logger.addHandler(error_handler)

# Blackjack-Helpers (aus Gambling-Projekt)
SUITS = ['♠', '♣', '♥', '♦']
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

def create_deck():
    deck = [(rank, suit) for suit in SUITS for rank in RANKS]
    random.shuffle(deck)
    return deck

def calculate_hand_value(hand):
    value = 0
    aces = 0
    for card in hand:
        rank = card[0]
        if rank in ['J', 'Q', 'K']:
            value += 10
        elif rank == 'A':
            aces += 1
        else:
            value += int(rank)
    for _ in range(aces):
        if value + 11 <= 21:
            value += 11
        else:
            value += 1
    return value

def get_user_money(user_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT money FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result:
            return Decimal(result[0])
    except Exception as e:
        print("DB error:", e)
    return Decimal(0)

def update_user_money(user_id, new_money):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET money = %s WHERE id = %s", (str(new_money), user_id))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("DB error:", e)

def get_username(user_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result:
            return result[0]
    except Exception as e:
        print("DB error:", e)
    return "unknown"

@app.route('/')
def index():
    # Session explizit löschen, damit niemand ohne FaceID eingeloggt bleibt
    session.pop('user_id', None)
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('user_id'):
        return redirect(url_for('index'))
    return redirect(url_for('blackjack'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    # Setze auch den Login-Status für den Client zurück
    client_id = request.remote_addr
    login_status.pop(client_id, None)
    return redirect(url_for('index'))

@app.route('/status')
def status():
    client_id = request.remote_addr
    # Setze session['logged_in'] erst hier, wenn login_status gesetzt ist
    if login_status.get(client_id):
        session['user_id'] = login_status[client_id]
        login_status.pop(client_id, None)
        return jsonify({'logged_in': True})
    return jsonify({'logged_in': bool(session.get('user_id'))})

@app.route('/register_face', methods=['POST'])
def register_face():
    print("Register Face: Request received")  # Debug
    data = request.get_json()
    if not data:
        print("Register Face: No data received")  # Debug
        return jsonify({'success': False, 'message': 'Keine Daten empfangen.'})
    if 'image' not in data:
        print("Register Face: No image in request")  # Debug
        return jsonify({'success': False, 'message': 'Kein Bild empfangen.'})
    if 'username' not in data:
        print("Register Face: No username in request")  # Debug
        return jsonify({'success': False, 'message': 'Kein Username empfangen.'})

    username = data['username']
    # Prüfe, ob Username schon existiert
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Username existiert bereits!'})
    except Exception as e:
        print(f"Register Face: DB-Fehler beim Username-Check: {e}")  # Debug
        return jsonify({'success': False, 'message': f'DB-Fehler: {str(e)}'})

    try:
        # Bilddaten dekodieren
        print("Register Face: Decoding image data")  # Debug
        if ',' in data['image']:
            image_data = data['image'].split(',')[1]
        else:
            image_data = data['image']
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            print("Register Face: Frame is None after decoding")  # Debug
            return jsonify({'success': False, 'message': 'Bild konnte nicht verarbeitet werden (frame is None).'})

        # Zusätzliche Prüfung: Ist das Bild schwarz oder leer?
        if np.sum(frame) == 0:
            print("Register Face: Frame is completely black or empty")  # Debug
            return jsonify({'success': False, 'message': 'Bild ist leer oder Kamera liefert kein Bild. Bitte Kamera prüfen und erneut versuchen.'})

        print("Register Face: Frame shape:", frame.shape)  # Debug

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

        # User in DB anlegen
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password, money) VALUES (%s, %s, %s)", (username, '', 100))
            conn.commit()
            user_id = cursor.lastrowid
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Register Face: DB-Fehler beim Anlegen: {e}")  # Debug
            return jsonify({'success': False, 'message': f'DB-Fehler beim Anlegen: {str(e)}'})

        # Gesicht speichern (Verzeichnis anlegen, falls nicht vorhanden)
        save_dir = os.path.join('assets', 'Good')
        if not os.path.exists(save_dir):
            print(f"Register Face: Creating directory {save_dir}")  # Debug
            os.makedirs(save_dir)
        filename = f"{user_id}.jpg"
        save_path = os.path.join(save_dir, filename)
        print(f"Register Face: Saving image to {save_path}")  # Debug
        success = cv2.imwrite(save_path, frame)
        if not success:
            print("Register Face: Error saving image")  # Debug
            return jsonify({'success': False, 'message': 'Fehler beim Speichern des Bildes.'})
        print("Register Face: Success!")  # Debug

        # Encoding nachladen
        encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
        known_encodings.append(encoding)
        known_user_ids.append(user_id)
        face_user_map[encoding.tobytes()] = user_id
        # Reload all encodings to ensure consistency
        reload_face_encodings()

        return jsonify({'success': True, 'message': 'Gesicht und Account erfolgreich registriert!'})
    except Exception as e:
        print(f"Register Face: Exception occurred: {e}")  # Debug
        return jsonify({'success': False, 'message': f'Fehler: {str(e)}'})


# Blackjack-Routen (angepasst: kein klassischer Login mehr, sondern session['user_id'] durch FaceID)
@app.route("/blackjack")
def blackjack():
    if 'user_id' not in session:
        return redirect(url_for("index"))
    user_id = session['user_id']
    money = get_user_money(user_id)
    if 'deck' not in session:
        session['deck'] = create_deck()
        session.modified = True
    return render_template("blackjack.html",
                         money=money,
                         player_hand=session.get('player_hand', []),
                         dealer_hand=session.get('dealer_hand', []),
                         game_over=session.get('game_over', False),
                         message=session.get('message', ""),
                         current_bet=session.get('current_bet', 10),
                         calculate_hand_value=calculate_hand_value)

@app.route("/blackjack/new", methods=["POST"])
def new_game():
    if 'user_id' not in session:
        return redirect(url_for("index"))
    user_id = session['user_id']
    money = get_user_money(user_id)
    username = get_username(user_id)
    try:
        bet_amount = Decimal(request.form.get('bet_amount', 10))
        if bet_amount < 1:
            session['message'] = "Minimum bet is $1!"
            return redirect(url_for('blackjack'))
    except Exception:
        session['message'] = "Invalid bet amount!"
        return redirect(url_for('blackjack'))
    if money < bet_amount:
        session['message'] = "Not enough money to place that bet!"
        session['game_over'] = True
        session.modified = True
        if username:
            error_logger.error(f"Insufficient funds - User: {username}, Attempted bet: ${bet_amount}, Balance: ${money}")
        return redirect(url_for('blackjack'))
    deck = create_deck()
    player_hand = [deck.pop() for _ in range(2)]
    dealer_hand = [deck.pop() for _ in range(2)]
    session['deck'] = deck
    session['player_hand'] = player_hand
    session['dealer_hand'] = dealer_hand
    session['game_over'] = False
    session['message'] = ''
    session['current_bet'] = bet_amount
    session.modified = True
    money -= bet_amount
    update_user_money(user_id, money)
    if username:
        game_logger.info(f"New game: {username} - Bet: ${bet_amount} - Balance: ${money}")
    if calculate_hand_value(player_hand) == 21:
        session['message'] = 'Blackjack! You win!'
        session['game_over'] = True
        winnings = bet_amount * Decimal('2.5')
        money += winnings
        update_user_money(user_id, money)
        session.modified = True
        if username:
            game_logger.info(f"Blackjack win: {username} - Bet: ${bet_amount} - Balance: ${money}")
    return redirect(url_for('blackjack'))

@app.route("/blackjack/hit")
def hit():
    if 'user_id' not in session:
        return redirect(url_for("index"))
    if session.get('game_over'):
        return redirect(url_for('blackjack'))
    user_id = session['user_id']
    username = get_username(user_id)
    bet_amount = session.get('current_bet', 10)
    deck = session['deck']
    player_hand = session['player_hand']
    player_hand.append(deck.pop())
    session['deck'] = deck
    session['player_hand'] = player_hand
    session.modified = True
    player_value = calculate_hand_value(player_hand)
    if player_value > 21:
        session['message'] = 'Bust! You lose!'
        session['game_over'] = True
        session.modified = True
        if username:
            error_logger.info(f"Player Bust - User: {username}, Bet: ${bet_amount}, Final Value: {player_value}")
    elif player_value == 21:
        session['message'] = 'You got 21!'
        session.modified = True
        if username:
            error_logger.info(f"Player got 21 - User: {username}, Bet: ${bet_amount}")
        return redirect(url_for('blackjack_stand'))
    return redirect(url_for('blackjack'))

@app.route("/blackjack/stand")
def blackjack_stand():
    if 'user_id' not in session:
        return redirect(url_for("index"))
    if session.get('game_over'):
        return redirect(url_for('blackjack'))
    user_id = session['user_id']
    money = get_user_money(user_id)
    username = get_username(user_id)
    bet_amount = session.get('current_bet', 10)
    deck = session['deck']
    dealer_hand = session['dealer_hand']
    while calculate_hand_value(dealer_hand) < 17:
        dealer_hand.append(deck.pop())
    session['deck'] = deck
    session['dealer_hand'] = dealer_hand
    session.modified = True
    player_value = calculate_hand_value(session['player_hand'])
    dealer_value = calculate_hand_value(dealer_hand)
    if dealer_value > 21:
        session['message'] = 'Dealer busts! You win!'
        money += bet_amount * 2
        if username:
            game_logger.info(f"Win: {username} - Bet: ${bet_amount} - Balance: ${money}")
    elif dealer_value > player_value:
        session['message'] = 'Dealer wins!'
        if username:
            game_logger.info(f"Loss: {username} - Bet: ${bet_amount} - Balance: ${money}")
    elif dealer_value < player_value:
        session['message'] = 'You win!'
        money += bet_amount * 2
        if username:
            game_logger.info(f"Win: {username} - Bet: ${bet_amount} - Balance: ${money}")
    else:
        session['message'] = 'Push! It\'s a tie!'
        money += bet_amount
        if username:
            game_logger.info(f"Tie: {username} - Bet: ${bet_amount} - Balance: ${money}")
    session['game_over'] = True
    session.modified = True
    update_user_money(user_id, money)
    return redirect(url_for('blackjack'))

@app.route("/leaderboard")
def leaderboard():
    if 'user_id' not in session:
        return redirect(url_for("index"))
    conn = get_db_connection()
    leaderboard_data = []
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT username, money 
                FROM users 
                ORDER BY money DESC 
                LIMIT 10
            """)
            leaderboard_data = cursor.fetchall()
            cursor.close()
            conn.close()
        except Exception as e:
            error_logger.error(f"Leaderboard error: {e}")
    user_id = session['user_id']
    user_rank = None
    user_money = get_user_money(user_id)
    username = get_username(user_id)
    if conn := get_db_connection():
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) + 1 
                FROM users 
                WHERE money > (
                    SELECT money 
                    FROM users 
                    WHERE id = %s
                )
            """, (user_id,))
            result = cursor.fetchone()
            if result:
                user_rank = result[0]
            cursor.close()
            conn.close()
        except Exception as e:
            error_logger.error(f"User rank error: {e}")
    return render_template(
        "leaderboard.html",
        leaderboard=leaderboard_data,
        user_rank=user_rank,
        user_money=user_money,
        username=username
    )

# Passe dashboard-Route an, leite direkt auf Blackjack weiter
# ENTFERNEN:
# @app.route('/dashboard')
# def dashboard():
#     if not session.get('user_id'):
#         return redirect(url_for('index'))
#     return redirect(url_for('blackjack'))

# Passe Logout an (löscht Session und Login-Status)
# ENTFERNEN:
# @app.route('/logout')
# def logout():
#     session.pop('user_id', None)
#     client_id = request.remote_addr
#     login_status.pop(client_id, None)
#     return redirect(url_for('index'))

# ...alle anderen FaceID- und Hilfsrouten bleiben wie gehabt...

if __name__ == '__main__':
    app.run(debug=True)
