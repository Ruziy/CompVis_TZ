import io
import base64
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw
import numpy as np
import cv2
import tempfile
import os
import torch
from yolov5 import YOLOv5
from ultralytics import YOLO
import pathlib
import platform

app = FastAPI()

# Определяем корректный путь в зависимости от операционной системы
plt = platform.system()
if plt == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# Define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_status = f"Using device: {device}"
print(device_status)

# Load models with specified device
model_yolov5x6 = YOLOv5(r"weights/best_yolov5x6.pt", device=device)
model_yolov5 = YOLOv5(r"weights/best_yolov5.pt", device=device)
model_yolov8 = YOLO(r"weights/yolov8n.pt").to(device)

# Minimum confidence threshold
CONFIDENCE_THRESHOLD = 0.3

def draw_bboxes(image: Image, boxes: list, labels: list) -> Image:
    """Draw bounding boxes with labels and confidence on the image."""
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box[:4]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), label, fill="red")
    return image

def video_stream(video_path):
    """Stream video with object detection and bounding box annotation."""
    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Обработка кадра YOLOv8
        results = model_yolov8.predict(frame, imgsz=640, save=False)
        bboxes = []
        labels = []
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box, conf, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
                    if conf >= CONFIDENCE_THRESHOLD:
                        bboxes.append([int(coord) for coord in box])
                        label = f"{conf:.2f}"
                        labels.append(label)

        # Рисуем рамки вокруг обнаруженных объектов
        for box, label in zip(bboxes, labels):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Кодируем кадр в JPEG формат
        success, encoded_frame = cv2.imencode('.jpg', frame)
        if not success:
            continue
        frame_bytes = encoded_frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video.release()
    os.remove(video_path)

@app.get("/", response_class=HTMLResponse)
async def main():
    return f"""
    <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f8f9fa;
                    color: #333;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    flex-direction: column;
                }}
                h2 {{
                    color: #444;
                }}
                form {{
                    text-align: center;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    background-color: #ffffff;
                    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                    width: 300px;
                }}
                select, input[type="file"], input[type="submit"] {{
                    width: 100%;
                    padding: 10px;
                    margin-top: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    font-size: 16px;
                }}
            </style>
        </head>
        <body>
            <h2>Загрузка для детекции объектов</h2>
            <p>{device_status}</p>
            <form action="/detect" method="post" enctype="multipart/form-data">
                <label for="input_type">Выберите тип файла:</label>
                <select name="input_type" id="input_type">
                    <option value="image">Изображение</option>
                    <option value="video">Видео</option>
                </select>
                <br><br>
                <label for="model">Выберите модель:</label>
                <select name="model" id="model">
                    <option value="yolov5">YOLOv5</option>
                    <option value="yolov5x6">YOLOv5x6</option>
                    <option value="yolov8">YOLOv8</option>
                </select>
                <br><br>
                <input name="file" type="file" required>
                <input type="submit" value="Обработать">
            </form>
        </body>
    </html>
    """

@app.post("/detect", response_class=HTMLResponse)
async def detect(file: UploadFile = File(...), model: str = Form(...), input_type: str = Form(...)):
    if input_type == "image":
        try:
            # Try to open and process the image
            image = Image.open(io.BytesIO(await file.read()))
            image_np = np.array(image)
            bboxes = []
            labels = []

            # Check model selection and apply respective prediction
            if model == "yolov5":
                results = model_yolov5.predict(image_np, size=640)
                for box, conf, cls in zip(results.xyxy[0], results.conf[0], results.cls[0]):
                    if conf >= CONFIDENCE_THRESHOLD:
                        bboxes.append([int(coord) for coord in box[:4]])
                        # label = f"Class {int(cls)}: {conf:.2f}"
                        # labels.append(label)

            elif model == "yolov5x6":
                results = model_yolov5x6.predict(image_np, size=640)
                for box, conf, cls in zip(results.xyxy[0], results.conf[0], results.cls[0]):
                    if conf >= CONFIDENCE_THRESHOLD:
                        bboxes.append([int(coord) for coord in box[:4]])
                        # label = f"Class {int(cls)}: {conf:.2f}"
                        # labels.append(label)

            elif model == "yolov8":
                results = model_yolov8.predict(image, imgsz=640, save=False)
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for box, conf, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
                            if conf >= CONFIDENCE_THRESHOLD:
                                bboxes.append([int(coord) for coord in box])
                                # label = f"Class {int(cls)}: {conf:.2f}"
                                # labels.append(label)

            # Draw bounding boxes with labels
            masked_image = draw_bboxes(image.copy(), bboxes, labels)

            # Convert to base64 and display
            buffered = io.BytesIO()
            masked_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return HTMLResponse(content=f"""
            <html>
                <head>
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            background-color: #f8f9fa;
                            color: #333;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            height: 100vh;
                            margin: 0;
                            flex-direction: column;
                        }}
                        h2 {{
                            color: #444;
                        }}
                        .container {{
                            text-align: center;
                            width: 80%;
                            max-width: 600px;
                            padding: 20px;
                            border: 1px solid #ddd;
                            border-radius: 8px;
                            background-color: #ffffff;
                            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                        }}
                        img {{
                            border: 3px solid #ddd;
                            border-radius: 8px;
                            max-width: 100%;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h2>Результат детекции:</h2>
                        <img src="data:image/jpeg;base64,{img_str}" alt="Detected Image"/>
                        <br><br>
                        <a href="/">Загрузить новое изображение</a>
                    </div>
                </body>
            </html>
            """)
        except Exception as e:
            # Handle exceptions and return an error message
            return HTMLResponse(content=f"""
            <html>
                <head>
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            background-color: #f8f9fa;
                            color: #333;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            height: 100vh;
                            margin: 0;
                            flex-direction: column;
                        }}
                        h2 {{
                            color: #444;
                        }}
                    </style>
                </head>
                <body>
                    <h2>Ошибка при обработке изображения:</h2>
                    <p>{str(e)}</p>
                    <br>
                    <a href="/">Загрузить другое изображение</a>
                </body>
            </html>
            """, status_code=400)

    elif input_type == "video":
        video_path = tempfile.mktemp(suffix=".mp4")
        with open(video_path, "wb") as buffer:
            buffer.write(await file.read())

        return StreamingResponse(video_stream(video_path), media_type="multipart/x-mixed-replace; boundary=frame")

