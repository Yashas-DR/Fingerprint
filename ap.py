from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import torch
from PIL import Image
import os
import sqlite3
import logging
from werkzeug.security import generate_password_hash, check_password_hash
import torchvision.transforms as transforms
import google.generativeai as genai
import io
from dotenv import load_dotenv
load_dotenv()
import gdown
MODEL_PATH = "fingerprint_blood_group_model.pkl"
MODEL_DRIVE_ID = "1AAmwHURkbS4vmfYNRtmXofRs4bB6T3JT"

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-pro-vision')

# SQLite DB connection
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

# Create users table if not exists
def init_db():
    with get_db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                fullname TEXT,
                email TEXT UNIQUE,
                username TEXT UNIQUE,
                password TEXT
            )
        ''')
        conn.commit()

init_db()

# CNN Model Definition
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(64 * 56 * 56, 512)
        self.fc2 = torch.nn.Linear(512, 8)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Download model from Google Drive if not found locally
if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model file from Google Drive...")
    gdown.download(id=MODEL_DRIVE_ID, output=MODEL_PATH, quiet=False)
    
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Verify image using Gemini
# Gemini API configuration
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY is not set in environment variables")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

def convert_bmp_to_jpg(bmp_path):
    jpg_path = os.path.splitext(bmp_path)[0] + ".jpg"
    with Image.open(bmp_path) as bmp_image:
        rgb_image = bmp_image.convert('RGB')
        rgb_image.save(jpg_path, 'JPEG')
    return jpg_path

def is_fingerprint_image(image_path):
    with Image.open(image_path) as img:
        response = gemini_model.generate_content(
            [
                "Determine if this is a fingerprint image. Answer strictly 'Yes' or 'No'.",
                img
            ],
            stream=False
        )
    return response.text.strip().lower().startswith("yes")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    ext = os.path.splitext(filename)[1].lower()
    if ext == ".bmp":
        file_path = convert_bmp_to_jpg(file_path)
    
    # Step 1: Gemini verification
    if not is_fingerprint_image(file_path):
        return render_template('error.html', message="Please upload a valid fingerprint image.")

    # Step 2: Actual prediction logic (as in your current code)
    img = Image.open(file_path).convert('RGB')
    img_tensor = data_transform(img).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img_tensor)
    predicted_blood_group = torch.argmax(prediction).item()
    blood_groups = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
    predicted_group_name = blood_groups[predicted_blood_group]
    return render_template('result.html', blood_group=predicted_group_name)

# Basic routes
@app.route('/')
def index(): return render_template('index.html')

@app.route('/home')
def home(): return render_template('home.html')

@app.route('/about')
def about(): return render_template('about.html')

@app.route('/Accurancy')
def Accurancy(): return render_template('chart.html')

@app.route('/team')
def team(): return render_template('team.html')

@app.route('/portfolio_details')
def portfolio_details(): return render_template('portfolio-details.html')

@app.route('/predict_blood_group')
def predict_blood_group():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))
    return render_template('login2.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            return redirect(url_for('predict_blood_group'))
        flash('Invalid username or password!', 'error')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']

        if password != confirmpassword:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (fullname, email, username, password) VALUES (?, ?, ?, ?)',
                         (fullname, email, username, hashed_password))
            conn.commit()
            flash('Account created successfully!', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
        finally:
            conn.close()

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        flash('Please log in to access your profile.', 'error')
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()

    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        username = request.form['username']

        if not fullname or not email or not username:
            flash('All fields are required.', 'error')
        else:
            try:
                conn.execute('''
                    UPDATE users SET fullname = ?, email = ?, username = ? WHERE id = ?
                ''', (fullname, email, username, user_id))
                conn.commit()
                flash('Profile updated successfully!', 'success')
            except Exception as e:
                logging.error(f"Profile update failed: {e}")
                flash('An error occurred.', 'error')
        conn.close()
        return redirect(url_for('profile'))

    return render_template('profile.html', user=user)

if __name__ == '__main__':
    app.run(debug=True)
