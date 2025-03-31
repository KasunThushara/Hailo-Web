from flask import Flask, Response, render_template_string, request, redirect, url_for
import socket
import pickle
import cv2
import threading
import os
import subprocess
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'hef', 'txt', 'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
latest_frame = None
detection_process = None
stream_active = False  # Flag to control when to show the stream

# UDP Socket setup for receiving frames
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(("0.0.0.0", 9999))
BUFFER_SIZE = 65536

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def receive_frames():
    global latest_frame
    while True:
        try:
            packet, _ = server_socket.recvfrom(BUFFER_SIZE)
            buffer = pickle.loads(packet)
            latest_frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error receiving frame: {e}")
            break

def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def run_detection(model_path, input_source, label_path):
    global detection_process, stream_active
    
    # Change directory and activate virtual environment
    base_cmd = "cd /home/pi/Hailo-Web-App && "
    base_cmd += "source env/bin/activate && "
    base_cmd += "cd Hailo-Web && "
    
    
    # Build the detection command
    if input_source.lower() == 'camera':
        detection_cmd = f"python3 client_object_detection.py -n {model_path} -i camera -l {label_path}"
    else:
        detection_cmd = f"python3 client_object_detection.py -n {model_path} -i {input_source} -l {label_path}"
    
    full_cmd = base_cmd + detection_cmd
    
    # Run the command in a new process
    detection_process = subprocess.Popen(full_cmd, shell=True, executable='/bin/bash')
    stream_active = True  # Activate the stream after starting detection

@app.route('/')
def index():
    global stream_active
    
    if stream_active:
        return """
        <html>
            <head>
                <title>Live Streaming</title>
            </head>
            <body>
                <h1>Live Streaming</h1>
                <img src="/video_feed" width="640" height="480">
                <br>
                <form action="/stop" method="POST">
                    <button type="submit">Stop Detection</button>
                </form>
            </body>
        </html>
        """
    else:
        return redirect(url_for('setup'))

@app.route('/video_feed')
def video_feed():
    if stream_active:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Stream not active", 404

@app.route('/stop', methods=['POST'])
def stop_detection():
    global detection_process, stream_active
    
    if detection_process:
        detection_process.terminate()
        detection_process = None
    
    stream_active = False
    return redirect(url_for('setup'))

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    global detection_process, stream_active
    
    if request.method == 'POST':
        # Get form data
        model_file = request.files['model']
        label_file = request.files['label']
        input_type = request.form['input_type']
        mp4_file = request.files.get('mp4_file')
        
        # Save uploaded files
        model_path = None
        label_path = None
        input_source = 'camera'
        
        if model_file and allowed_file(model_file.filename):
            model_filename = secure_filename(model_file.filename)
            model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)
            model_file.save(model_path)
        
        if label_file and allowed_file(label_file.filename):
            label_filename = secure_filename(label_file.filename)
            label_path = os.path.join(app.config['UPLOAD_FOLDER'], label_filename)
            label_file.save(label_path)
        
        if input_type == 'mp4' and mp4_file and allowed_file(mp4_file.filename):
            mp4_filename = secure_filename(mp4_file.filename)
            input_source = os.path.join(app.config['UPLOAD_FOLDER'], mp4_filename)
            mp4_file.save(input_source)
        
        # Run detection if all required files are provided
        if model_path and label_path:
            run_detection(model_path, input_source, label_path)
            return redirect(url_for('index'))
    
    return render_template_string('''
    <html>
        <head>
            <title>Detection Setup</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .form-group {
                    margin-bottom: 15px;
                }
                label {
                    display: block;
                    margin-bottom: 5px;
                    font-weight: bold;
                }
                input[type="file"], input[type="radio"] {
                    margin-bottom: 10px;
                }
                button {
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 15px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    margin-right: 10px;
                }
                button:hover {
                    background-color: #45a049;
                }
            </style>
        </head>
        <body>
            <h1>Detection Setup</h1>
            <form method="POST" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="model">HEF Model File:</label>
                    <input type="file" id="model" name="model" accept=".hef" required>
                </div>
                <div class="form-group">
                    <label for="label">Label File:</label>
                    <input type="file" id="label" name="label" accept=".txt" required>
                </div>
                <div class="form-group">
                    <label>Input Source:</label>
                    <div>
                        <input type="radio" id="camera" name="input_type" value="camera" checked>
                        <label for="camera" style="display: inline;">Camera</label>
                    </div>
                    <div>
                        <input type="radio" id="mp4" name="input_type" value="mp4">
                        <label for="mp4" style="display: inline;">MP4 File</label>
                    </div>
                </div>
                <div id="mp4_file_container" style="display:none;" class="form-group">
                    <label for="mp4_file">MP4 File:</label>
                    <input type="file" id="mp4_file" name="mp4_file" accept=".mp4">
                </div>
                <div>
                    <button type="submit">Run Detection</button>
                </div>
            </form>
            <script>
                document.querySelectorAll('input[name="input_type"]').forEach(radio => {
                    radio.addEventListener('change', function() {
                        document.getElementById('mp4_file_container').style.display = 
                            this.value === 'mp4' ? 'block' : 'none';
                    });
                });
            </script>
        </body>
    </html>
    ''')

if __name__ == '__main__':
    receive_thread = threading.Thread(target=receive_frames)
    receive_thread.daemon = True
    receive_thread.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
