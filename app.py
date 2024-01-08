import io
import time
import numpy as np
from flask import Flask, render_template, Response
import cv2
import torch
from PIL import Image
import csv
from queue import Queue
from flask import jsonify
import pandas as pd

model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Dell/Documents/AMIELIA FYP/TEST/yolov5/runs/train/exp15/weights/best.pt', force_reload=True)

# Initialize Flask Application
app = Flask(__name__)

recent_item = ""
recent_status = ""


def detection():
    cap = cv2.VideoCapture(0)
    with open('safety_data.csv', mode='w', newline='') as file:
        fieldnames = ['class', 'status']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                ret, buffer = cv2.imencode('.jpeg', frame)
                frame = buffer.tobytes()
                img = Image.open(io.BytesIO(frame))
                results = model(img, size=640)
                row = results.pandas().xyxy[0]['name'].tolist()
                classes = []
                classes.append(row)
                safe = ["hard hat", "hi-vis vest"]
                semisafe1 = ["hard hat", "no hi-vis vest"]
                semisafe2 = ["no hard hat", "hi-vis vest"]
                unsafe = ["no hard hat", "no hi-vis vest"]
                for item in classes:
                    if all(keyword in item for keyword in safe):
                        status = "Very Safe"
                    elif all(keyword in item for keyword in semisafe1):
                        status = "Semi Safe"
                    elif all(keyword in item for keyword in semisafe2):
                        status = "Semi Safe"
                    elif all(keyword in item for keyword in unsafe):
                        status = "Not Safe"
                    else:
                        status = "No detection"

                    global recent_item, recent_status
                    recent_item = item
                    recent_status = status
                    writer.writerow({'class': item, 'status': status})

                img = np.squeeze(results.render())
                img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                break

            frame = cv2.imencode('.jpeg', img_BGR)[1].tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/report')
def report():
    return render_template('report.html')


@app.route('/video_feed')
def video_feed():
    return Response(detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_data')
def update_data():
    global recent_item, recent_status
    data = {
        'status': recent_status,
        'item': recent_item
    }
    return jsonify(data)

@app.route('/get_data')
def get_data():
    data = {} 
    df = pd.read_csv('safety_data.csv')
    filtered_df = df[df['status'].isin(['Not Safe', 'Semi Safe', 'Very Safe'])]
    counts = filtered_df['status'].value_counts()
    for label, count in counts.items():
        data[label] = count

    return jsonify(data)

@app.route('/get_items')
def get_items():
    df = pd.read_csv('safety_data.csv')
    data = {
        'HH': int(df['class'].apply(lambda x: 'hard hat' in x).sum()),
        'NHH': int(df['class'].apply(lambda x: 'no hard hat' in x).sum()),
        'HV': int(df['class'].apply(lambda x: 'hi-vis vest' in x).sum()),
        'NHV': int(df['class'].apply(lambda x: 'no hi-vis vest' in x).sum()),
        'VS' : int(df['status'].apply(lambda x: 'Very Safe' in x).sum()),
        'SS' : int(df['status'].apply(lambda x: 'Semi Safe' in x).sum()),
        'NS' : int(df['status'].apply(lambda x: 'Not Safe' in x).sum())
    }
        
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
