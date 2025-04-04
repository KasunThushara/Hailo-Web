from flask import Flask, Response, render_template, request, redirect, url_for
import socket
import pickle
import cv2
import threading
import os
import subprocess
import signal
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'hef', 'txt', 'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
latest_frame = None
current_process = None
stream_active = False
current_mode = None  # 'object' or 'pose'

# UDP Socket setup for receiving frames
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(("0.0.0.0", 9999))
server_socket.settimeout(1.0)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def receive_frames():
    global latest_frame
    while True:
        try:
            packet, _ = server_socket.recvfrom(65536)
            buffer = pickle.loads(packet)
            latest_frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"Frame receive error: {e}")
            break

def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None and stream_active:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.01)  # ~30 FPS

def run_detection(model_path, input_source, label_path):
    global current_process
    
    # Change directory and activate virtual environment
    base_cmd = "cd /home/pi/Hailo-Web-App && "
    base_cmd += "source env/bin/activate && "
    base_cmd += "cd Hailo-Web && "
    
    if input_source.lower() == 'camera':
        detection_cmd = f"python3 client_object_detection.py -n {model_path} -i camera -l {label_path}"
    else:
        detection_cmd = f"python3 client_object_detection.py -n {model_path} -i {input_source} -l {label_path}"
    
    full_cmd = base_cmd + detection_cmd
    
    # Run the command in a new process
    current_process = subprocess.Popen(
        full_cmd, 
        shell=True, 
        executable='/bin/bash',
        preexec_fn=os.setsid
    )

def run_pose_estimation(model_path, input_source):
    global current_process
    
    # Change directory and activate virtual environment
    base_cmd = "cd /home/pi/Hailo-Web-App && "
    base_cmd += "source env/bin/activate && "
    base_cmd += "cd Hailo-Web && "
    
    if input_source.lower() == 'camera':
        cmd = f"python3 client_pose_estimation.py -n {model_path} -i 0"  # Camera index 0
    else:
        cmd = f"python3 client_pose_estimation.py -n {model_path} -i {input_source}"
    
    full_cmd = base_cmd + cmd
    
    # Run in a new process group for proper termination
    current_process = subprocess.Popen(
        full_cmd, 
        shell=True, 
        executable='/bin/bash',
        preexec_fn=os.setsid
    )

def stop_current_process():
    global current_process, stream_active
    if current_process:
        os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
        current_process = None
    stream_active = False

@app.route('/')
def index():
    if stream_active:
        mode_title = "Object Detection" if current_mode == 'object' else "Pose Estimation"
        return render_template('live_view.html', mode_title=mode_title)
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/setup', methods=['GET'])
def setup_mode():
    global current_mode
    mode = request.args.get('mode')
    if mode not in ['object', 'pose']:
        return redirect(url_for('index'))
    
    current_mode = mode
    return redirect(url_for('setup_config'))

@app.route('/setup_config', methods=['GET', 'POST'])
def setup_config():
    global current_mode, stream_active
    
    if request.method == 'POST':
        # Check if back button was pressed
        if 'back' in request.form:
            return redirect(url_for('index'))
        
        # Handle form submission based on mode
        if current_mode == 'object':
            model_file = request.files.get('model')
            label_file = request.files.get('label')
            input_type = request.form.get('input_type')
            mp4_file = request.files.get('mp4_file')
            
            # Validate inputs
            if not model_file or not label_file:
                return "Both model and label files are required", 400
                
            if not allowed_file(model_file.filename) or not allowed_file(label_file.filename):
                return "Invalid file type", 400
                
            # Save uploaded files
            model_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                    secure_filename(model_file.filename))
            model_file.save(model_path)
            
            label_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                    secure_filename(label_file.filename))
            label_file.save(label_path)
            
            # Determine input source
            input_source = 'camera'
            if input_type == 'mp4' and mp4_file and allowed_file(mp4_file.filename):
                input_source = os.path.join(app.config['UPLOAD_FOLDER'],
                                          secure_filename(mp4_file.filename))
                mp4_file.save(input_source)
            
            # Start detection
            run_detection(model_path, input_source, label_path)
            
        elif current_mode == 'pose':
            model_file = request.files.get('model')
            input_type = request.form.get('input_type')
            mp4_file = request.files.get('mp4_file')
            
            # Validate inputs
            if not model_file:
                return "Model file is required", 400
                
            if not allowed_file(model_file.filename):
                return "Invalid model file type", 400
                
            # Save model file
            model_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                    secure_filename(model_file.filename))
            model_file.save(model_path)
            
            # Determine input source
            input_source = 'camera'
            if input_type == 'mp4' and mp4_file and allowed_file(mp4_file.filename):
                input_source = os.path.join(app.config['UPLOAD_FOLDER'],
                                          secure_filename(mp4_file.filename))
                mp4_file.save(input_source)
            
            # Start pose estimation
            run_pose_estimation(model_path, input_source)
        
        stream_active = True
        return redirect(url_for('index'))
    
    # Render appropriate setup form based on mode
    if current_mode == 'object':
        return render_template('object_setup.html')
    else:  # pose estimation
        return render_template('pose_setup.html')

@app.route('/stop', methods=['POST'])
def stop():
    stop_current_process()
    return redirect(url_for('index'))

if __name__ == '__main__':
    frame_thread = threading.Thread(target=receive_frames)
    frame_thread.daemon = True
    frame_thread.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
