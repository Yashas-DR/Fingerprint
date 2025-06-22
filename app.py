from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import torch
from PIL import Image
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import torchvision.transforms as transforms
from flask_wtf import Form
"""
from wtforms import TextField, BooleanField
from wtforms.validators import Required"""

app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path for saving uploads
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Database connection
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

# Create users table if not exists
def init_db():
    with get_db_connection() as conn:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, fullname TEXT, email TEXT UNIQUE, username TEXT UNIQUE, password TEXT)')
        conn.commit()

init_db()

# Define your model architecture
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(64 * 56 * 56, 512)
        self.fc2 = torch.nn.Linear(512, 8)  # Assuming 8 classes for blood groups

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model and load the state dictionary
model = SimpleCNN()
model.load_state_dict(torch.load('fingerprint_blood_group_model.pkl', map_location=torch.device('cpu')))
model.eval()

# Define the transformation for the input image
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict blood group route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save file to upload folder
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load image, apply transformations
        img = Image.open(file_path).convert('RGB')
        img_tensor = data_transform(img).unsqueeze(0)  # Add batch dimension

        # Perform prediction
        with torch.no_grad():
            prediction = model(img_tensor)

        predicted_blood_group = torch.argmax(prediction).item()
        blood_groups = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
        predicted_group_name = blood_groups[predicted_blood_group]

        # Redirect to result page with the predicted blood group
        return render_template('result.html', blood_group=predicted_group_name)

# Routes for different pages
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/Accurancy')
def Accurancy():
    return render_template('chart.html')


@app.route('/team')
def team():
    return render_template('team.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            # flash('Login successful!', 'success')
            return redirect(url_for('predict_blood_group'))  # Or redirect to profile
        else:
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

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (fullname, email, username, password) VALUES (?, ?, ?, ?)',
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


import logging

# Set up basic logging configuration
logging.basicConfig(level=logging.DEBUG)


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        flash('Please log in to access your profile.', 'error')
        return redirect(url_for('login'))

    conn = get_db_connection()
    user_id = session['user_id']

    # Check if the connection is valid
    if conn is None:
        logging.error("Database connection failed!")
        flash('Database connection failed!', 'error')
        return redirect(url_for('login'))

    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()

    if user is None:
        logging.error(f"No user found with id {user_id}")
        flash('User not found.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        username = request.form['username']

        # Log the received form data for debugging
        logging.debug(f"Received data: fullname={fullname}, email={email}, username={username}")

        # Validate form data (add more validation as needed)
        if not fullname or not email or not username:
            flash('All fields are required.', 'error')
            return redirect(url_for('profile'))

        try:
            # Update the database with the new information
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET fullname = ?, email = ?, username = ? WHERE id = ?
            ''', (fullname, email, username, user_id))
            conn.commit()
            conn.close()
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('profile'))
        except Exception as e:
            logging.error(f"Error updating profile: {e}")
            flash('An error occurred while updating your profile. Please try again.', 'error')
            return redirect(url_for('profile'))

    return render_template('profile.html', user=user)

@app.route('/portfolio_details')
def portfolio_details():
    return render_template('portfolio-details.html')

	

@app.route('/predict_blood_group')
def predict_blood_group():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))
    return render_template('login2.html')

if __name__ == '__main__':
    app.run(debug=True)
