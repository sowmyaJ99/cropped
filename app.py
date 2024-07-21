from flask import Flask, request, redirect, url_for, send_from_directory, render_template
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

# Add YOLOv5 directory to path (if needed)
import sys
sys.path.append('yolov5')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CROPPED_FOLDER'] = 'cropped_objects'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB limit

def detect_objects_yolo(frame, model, device):
    img = letterbox(frame, 640, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB and transpose to (C, H, W)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=False)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    detected_objects = []

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape[:2]).round()
            labels = det[:, -1].cpu().numpy()
            label_names = [model.names[int(label)] for label in labels]

            for i, (label_name, *xyxy) in enumerate(zip(label_names, *det[:, :4].cpu().numpy().T)):
                xyxy = [int(x) for x in xyxy]
                x1, y1, x2, y2 = xyxy
                cropped_img = frame[y1:y2, x1:x2]

                detected_objects.append(cropped_img)

    return detected_objects

@app.route('/')
def index():
    # If 'folder' and 'cropped_files' are not in the session, they will be empty
    folder = request.args.get('folder', '')
    cropped_files = []
    if folder and os.path.exists(folder):
        cropped_files = os.listdir(folder)
    return render_template('index.html', folder=folder, cropped_files=cropped_files)

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = 'video.mp4'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        cropped_folder = process_video(file_path)
        if cropped_folder:
            return redirect(url_for('index', folder=cropped_folder))
        else:
            return 'No objects detected or an error occurred.'

@app.route('/cropped_objects/<filename>')
def display_cropped(filename):
    return send_from_directory(app.config['CROPPED_FOLDER'], filename)

def process_video(video_path):
    output_folder = os.path.join(app.config['CROPPED_FOLDER'])
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 10)  # Process every 10 seconds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend('yolov5s.pt', device=device)
    model.eval()

    count = 0
    frame_number = 0

    print(f"Processing frames from {video_path}...")
    pbar = tqdm(total=total_frames)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % frame_interval == 0:
            detected_objects = detect_objects_yolo(frame, model, device)
            for i, obj in enumerate(detected_objects):
                save_path = os.path.join(output_folder, f"frame_{frame_number:04d}_obj_{i:02d}.jpg")
                cv2.imwrite(save_path, obj)

        frame_number += 1
        pbar.update(1)
        pbar.set_description(f"Frames processed: {frame_number}/{total_frames}")

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()

    if os.listdir(output_folder):
        return output_folder
    else:
        return None

if __name__ == '__main__':
    app.run(debug=True)
