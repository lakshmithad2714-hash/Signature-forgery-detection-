from flask import Flask, render_template, request, jsonify
import sqlite3, os, cv2
import numpy as np
from tensorflow.keras.models import load_model

# ================= APP SETUP =================
app = Flask(__name__)

DATABASE = "database.db"
UPLOAD_FOLDER = "uploads"
DATASET_PATH = "dataset"
MODEL_PATH = "signature_model.h5"

IMG_SIZE = 128

# ================= SAFE FOLDERS =================
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "genuine"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "forged"), exist_ok=True)

# ================= LOAD MODEL =================
model = load_model(MODEL_PATH)

# ================= DATABASE =================
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            result TEXT
        )
    """)
    db.commit()

init_db()

# ================= IMAGE PREPROCESS =================
def preprocess_image(path):
    img = cv2.imread(path, 0)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# ================= HOME =================
@app.route("/")
@app.route("/test")
def home():
    return render_template("ai_dashboard.html")

# ================= PREDICTION API =================
@app.route("/test", methods=["POST"])
def predict():

    file = request.files.get("signature")

    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = preprocess_image(filepath)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # ===== MODEL PREDICTION =====
    pred_probs = model.predict(img)[0]

    print("Prediction probabilities:", pred_probs)

    pred = int(np.argmax(pred_probs))
    confidence = round(float(pred_probs[pred]) * 100, 2)

    print("Predicted class:", pred)
    print("Confidence:", confidence)

    result = "Genuine" if pred == 0 else "Forged"

    # ===== SAVE HISTORY =====
    db = get_db()
    db.execute(
        "INSERT INTO history (image_name, result) VALUES (?, ?)",
        (file.filename, result)
    )
    db.commit()

    return jsonify({
        "result": result,
        "accuracy": confidence
    })

# ================= HISTORY =================
@app.route("/history")
def history():
    db = get_db()
    records = db.execute("SELECT * FROM history").fetchall()
    return render_template("history.html", records=records)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)